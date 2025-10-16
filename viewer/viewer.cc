// FWMC Brain Viewer: real-time 3D visualization of the procedural brain.
// Shows the brain SDF as a lit point cloud with orbit camera controls.
// Cross-platform: Windows, Linux, macOS.

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "tissue/voxel_grid.h"
#include "tissue/brain_sdf.h"
#include "tissue/neural_field.h"
#include "core/neuron_array.h"
#include "core/izhikevich.h"
#include "core/synapse_table.h"
#include "core/stdp.h"
#include "core/motor_output.h"
#include "core/proprioception.h"
#include "core/cpg.h"
#include "core/brain_spec_loader.h"
#include "core/gap_junctions.h"
#include "core/intrinsic_homeostasis.h"
#include "core/cell_types.h"
#include "core/sim_features.h"
#include "core/spike_frequency_adaptation.h"
#include "core/structural_plasticity.h"
#include "core/temperature.h"
#include "core/inhibitory_plasticity.h"
#include "core/neuromodulator_effects.h"
#include "core/nmda.h"
#include "core/calcium_plasticity.h"
#include "core/rate_monitor.h"
#include "tcp_server.h"
#include "body_viewport.h"

using namespace fwmc;
using namespace mechabrain;

// Point vertex: position + color + size + region
struct PointVertex {
  float x, y, z;
  float r, g, b;
  float size;
  int region;
};

// ---------------------------------------------------------------------------
// FlyWire-based connectome builder.
//
// Maps the 13 SDF viewer regions to the 7 drosophila_full.brain spec regions,
// then wires up connectivity using the biologically-grounded density, weight,
// and neurotransmitter distributions from the FlyWire connectome model.
//
// Density is scaled to preserve out-degree per neuron:
//   actual_density = spec_density * spec_n / viewer_n   (capped at 0.5)
// This keeps the circuit dynamics independent of the viewer's point spacing.
// ---------------------------------------------------------------------------

// SDF region index (0-12) → drosophila_full.brain region name
static const char* kSDFToSpecName[13] = {
  "protocerebrum",      // 0 central_brain
  "optic_lobe",         // 1 optic_lobe_L
  "optic_lobe",         // 2 optic_lobe_R
  "mushroom_body",      // 3 mb_calyx_L
  "mushroom_body",      // 4 mb_calyx_R
  "mushroom_body",      // 5 mb_lobe_L
  "mushroom_body",      // 6 mb_lobe_R
  "antennal_lobe",      // 7 antennal_lobe_L
  "antennal_lobe",      // 8 antennal_lobe_R
  "central_complex",    // 9 central_complex
  "lateral_horn",       // 10 lateral_horn_L
  "lateral_horn",       // 11 lateral_horn_R
  "ventral_nerve_cord", // 12 sez / VNC
};

static SynapseTable BuildFlyWireConnectome(const std::vector<PointVertex>& pts,
                                            size_t n,
                                            const BrainSpec& spec) {
  // Helper: find spec region index by name
  auto find_spec = [&](const std::string& name) -> int {
    for (int i = 0; i < (int)spec.regions.size(); ++i)
      if (spec.regions[i].name == name) return i;
    return -1;
  };

  // Build SDF→spec index map
  int sdf_to_spec[13];
  for (int i = 0; i < 13; ++i)
    sdf_to_spec[i] = find_spec(kSDFToSpecName[i]);

  // Group viewer neurons by spec region
  int n_spec = static_cast<int>(spec.regions.size());
  std::vector<std::vector<uint32_t>> by_spec(n_spec);
  for (size_t i = 0; i < n; ++i) {
    int sdf_reg = pts[i].region;
    int sr = (sdf_reg >= 0 && sdf_reg < 13) ? sdf_to_spec[sdf_reg] : -1;
    if (sr >= 0) by_spec[sr].push_back(static_cast<uint32_t>(i));
  }

  std::mt19937 rng(spec.seed);
  std::uniform_real_distribution<float> coin(0.0f, 1.0f);

  std::vector<uint32_t> pre_vec, post_vec;
  std::vector<float>    w_vec;
  std::vector<uint8_t>  nt_vec;

  // Pick NT for a synapse based on a region's nt_distribution
  auto pick_nt = [&](const RegionSpec& reg) -> uint8_t {
    if (reg.nt_distribution.empty())
      return static_cast<uint8_t>(reg.default_nt);
    float roll = coin(rng), cum = 0.0f;
    for (auto& ntf : reg.nt_distribution) {
      cum += ntf.fraction;
      if (roll < cum) return static_cast<uint8_t>(ntf.nt);
    }
    return static_cast<uint8_t>(reg.default_nt);
  };

  // Internal connections — O(n * out_degree), preserves average out-degree.
  // out_degree = spec_density * spec_n_neurons (real expected fan-out per neuron)
  //
  // For bilateral regions (SEZ, lateral_horn_L/R), connections are mirrored:
  // each L→L synapse gets a matching R→R synapse with the same weight.
  // This enforces the L/R symmetry real descending circuits have, preventing
  // random connectivity from creating a turn bias in the motor output.
  constexpr float kMidlineX = 250.0f;
  int sez_spec = find_spec("sez");

  for (int r = 0; r < n_spec; ++r) {
    auto& bucket = by_spec[r];
    if (bucket.size() < 2) continue;
    const auto& reg = spec.regions[r];

    int out_degree = std::max(1, static_cast<int>(std::round(
        reg.internal_density * static_cast<float>(reg.n_neurons))));
    out_degree = std::min(out_degree, static_cast<int>(bucket.size()) - 1);

    std::normal_distribution<float> wdist(spec.global_weight_mean,
                                          spec.global_weight_std);

    // For SEZ: split into L/R halves and wire each half independently
    // with identical random seeds so both sides get mirror-symmetric wiring.
    if (r == sez_spec && bucket.size() >= 4) {
      std::vector<uint32_t> left_neurons, right_neurons;
      for (uint32_t idx : bucket) {
        if (pts[idx].x < kMidlineX) left_neurons.push_back(idx);
        else                        right_neurons.push_back(idx);
      }
      // Wire each half with the same pattern (same out_degree, same weight draws).
      // Use separate RNG copies seeded identically for mirror symmetry.
      auto rng_copy = rng;  // snapshot RNG state
      auto wire_half = [&](std::vector<uint32_t>& half, std::mt19937& gen) {
        if (half.size() < 2) return;
        int deg = std::min(out_degree, static_cast<int>(half.size()) - 1);
        std::uniform_int_distribution<size_t> pick_h(0, half.size() - 1);
        std::normal_distribution<float> wd(spec.global_weight_mean,
                                           spec.global_weight_std);
        for (size_t ni = 0; ni < half.size(); ++ni) {
          for (int c = 0; c < deg; ++c) {
            uint32_t post = half[pick_h(gen)];
            if (post == half[ni]) continue;
            pre_vec.push_back(half[ni]);
            post_vec.push_back(post);
            w_vec.push_back(std::max(0.1f, wd(gen)));
            nt_vec.push_back(pick_nt(reg));
          }
        }
      };
      auto rng_left = rng_copy;
      auto rng_right = rng_copy;
      wire_half(left_neurons, rng_left);
      wire_half(right_neurons, rng_right);
      // Advance main RNG past this region
      rng = rng_left;
    } else {
      // Non-bilateral regions: wire randomly as before.
      std::uniform_int_distribution<size_t> pick(0, bucket.size() - 1);
      for (uint32_t pre : bucket) {
        for (int c = 0; c < out_degree; ++c) {
          uint32_t post = bucket[pick(rng)];
          if (post == pre) continue;
          pre_vec.push_back(pre);
          post_vec.push_back(post);
          w_vec.push_back(std::max(0.1f, wdist(rng)));
          nt_vec.push_back(pick_nt(reg));
        }
      }
    }
  }

  // Inter-region projections from the spec (AL→MB, AL→LH, MB→CX, etc.)
  // O(n_src * out_degree), where out_degree = density * spec_n_to_neurons.
  for (auto& proj : spec.projections) {
    int from_r = find_spec(proj.from_region);
    int to_r   = find_spec(proj.to_region);
    if (from_r < 0 || to_r < 0) continue;

    auto& src = by_spec[from_r];
    auto& dst = by_spec[to_r];
    if (src.empty() || dst.empty()) continue;

    int out_degree = std::max(1, static_cast<int>(std::round(
        proj.density * static_cast<float>(spec.regions[to_r].n_neurons))));
    out_degree = std::min(out_degree, static_cast<int>(dst.size()));

    std::normal_distribution<float> wdist(proj.weight_mean, proj.weight_std);
    std::uniform_int_distribution<size_t> pick_dst(0, dst.size() - 1);

    for (uint32_t pre : src) {
      for (int c = 0; c < out_degree; ++c) {
        pre_vec.push_back(pre);
        post_vec.push_back(dst[pick_dst(rng)]);
        w_vec.push_back(std::max(0.1f, wdist(rng)));
        nt_vec.push_back(static_cast<uint8_t>(proj.nt_type));
      }
    }
  }

  SynapseTable table;
  table.BuildFromCOO(n, pre_vec, post_vec, w_vec, nt_vec);
  printf("FlyWire connectome: %zu neurons, %zu synapses (drosophila_full.brain)\n",
         n, table.Size());
  return table;
}

// Assign FlyWire cell types to viewer neurons from spec distributions.
static void AssignFlyWireCellTypes(NeuronArray& neurons,
                                   const std::vector<PointVertex>& pts,
                                   const BrainSpec& spec) {
  auto find_spec = [&](const std::string& name) -> int {
    for (int i = 0; i < (int)spec.regions.size(); ++i)
      if (spec.regions[i].name == name) return i;
    return -1;
  };

  int sdf_to_spec[13];
  for (int i = 0; i < 13; ++i)
    sdf_to_spec[i] = find_spec(kSDFToSpecName[i]);

  int n_spec = static_cast<int>(spec.regions.size());
  std::vector<std::vector<size_t>> by_spec(n_spec);
  for (size_t i = 0; i < neurons.n; ++i) {
    int sdf_reg = pts[i].region;
    int sr = (sdf_reg >= 0 && sdf_reg < 13) ? sdf_to_spec[sdf_reg] : -1;
    if (sr >= 0) by_spec[sr].push_back(i);
  }

  std::mt19937 rng(spec.seed + 99);
  for (int r = 0; r < n_spec; ++r) {
    auto& bucket = by_spec[r];
    if (bucket.empty()) continue;
    std::shuffle(bucket.begin(), bucket.end(), rng);
    const auto& reg = spec.regions[r];
    size_t assigned = 0;
    for (auto& ctf : reg.cell_types) {
      size_t n_this = static_cast<size_t>(ctf.fraction * bucket.size());
      n_this = std::min(n_this, bucket.size() - assigned);
      for (size_t j = 0; j < n_this; ++j)
        neurons.type[bucket[assigned + j]] = static_cast<uint8_t>(ctf.type);
      assigned += n_this;
    }
  }
}

// Simple 4x4 matrix math (avoids GLM dependency)
struct Mat4 {
  float m[16] = {};
  static Mat4 Identity() {
    Mat4 r;
    r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
    return r;
  }
  static Mat4 Perspective(float fov_rad, float aspect, float z_near, float z_far) {
    Mat4 r;
    float f = 1.0f / std::tan(fov_rad * 0.5f);
    r.m[0] = f / aspect;
    r.m[5] = f;
    r.m[10] = (z_far + z_near) / (z_near - z_far);
    r.m[11] = -1.0f;
    r.m[14] = (2.0f * z_far * z_near) / (z_near - z_far);
    return r;
  }
  static Mat4 LookAt(float ex, float ey, float ez,
                     float cx, float cy, float cz,
                     float ux, float uy, float uz) {
    float fx = cx - ex, fy = cy - ey, fz = cz - ez;
    float fl = std::sqrt(fx*fx + fy*fy + fz*fz);
    fx /= fl; fy /= fl; fz /= fl;
    float rx = fy*uz - fz*uy, ry = fz*ux - fx*uz, rz = fx*uy - fy*ux;
    float rl = std::sqrt(rx*rx + ry*ry + rz*rz);
    rx /= rl; ry /= rl; rz /= rl;
    ux = ry*fz - rz*fy; uy = rz*fx - rx*fz; uz = rx*fy - ry*fx;
    Mat4 r;
    r.m[0]=rx; r.m[4]=ry; r.m[8]=rz;
    r.m[1]=ux; r.m[5]=uy; r.m[9]=uz;
    r.m[2]=-fx; r.m[6]=-fy; r.m[10]=-fz;
    r.m[12]=-(rx*ex+ry*ey+rz*ez);
    r.m[13]=-(ux*ex+uy*ey+uz*ez);
    r.m[14]=(fx*ex+fy*ey+fz*ez);
    r.m[15]=1.0f;
    return r;
  }
};

// Orbit camera
struct OrbitCamera {
  float azimuth = 0.3f;
  float elevation = 0.4f;
  float radius = 500.0f;
  float target_x = 250.0f;
  float target_y = 150.0f;
  float target_z = 100.0f;

  void Eye(float& ex, float& ey, float& ez) const {
    ex = target_x + radius * std::cos(elevation) * std::sin(azimuth);
    ey = target_y + radius * std::sin(elevation);
    ez = target_z + radius * std::cos(elevation) * std::cos(azimuth);
  }

  Mat4 ViewMatrix() const {
    float ex, ey, ez;
    Eye(ex, ey, ez);
    return Mat4::LookAt(ex, ey, ez, target_x, target_y, target_z, 0, 1, 0);
  }

  void Reset() {
    azimuth = 0.3f; elevation = 0.4f; radius = 500.0f;
    target_x = 250.0f; target_y = 150.0f; target_z = 100.0f;
  }
};

// Shader compilation helper
static GLuint CompileShader(GLenum type, const char* src) {
  GLuint s = glCreateShader(type);
  glShaderSource(s, 1, &src, nullptr);
  glCompileShader(s);
  GLint ok;
  glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    char log[512];
    glGetShaderInfoLog(s, 512, nullptr, log);
    fprintf(stderr, "Shader error: %s\n", log);
  }
  return s;
}

static GLuint LinkProgram(GLuint vs, GLuint fs) {
  GLuint p = glCreateProgram();
  glAttachShader(p, vs);
  glAttachShader(p, fs);
  glBindAttribLocation(p, 0, "aPos");
  glBindAttribLocation(p, 1, "aColor");
  glBindAttribLocation(p, 2, "aSize");
  glLinkProgram(p);
  GLint ok;
  glGetProgramiv(p, GL_LINK_STATUS, &ok);
  if (!ok) {
    char log[512];
    glGetProgramInfoLog(p, 512, nullptr, log);
    fprintf(stderr, "Link error: %s\n", log);
  }
  glDeleteShader(vs);
  glDeleteShader(fs);
  return p;
}

// Region colors (distinct hues for each brain region)
static void RegionColor(int region, float& r, float& g, float& b) {
  static const float colors[][3] = {
    {0.9f, 0.7f, 0.5f},  // central_brain: warm beige
    {0.2f, 0.6f, 0.9f},  // optic_lobe_L: blue
    {0.2f, 0.6f, 0.9f},  // optic_lobe_R: blue
    {0.9f, 0.3f, 0.5f},  // mb_calyx_L: pink
    {0.9f, 0.3f, 0.5f},  // mb_calyx_R: pink
    {0.8f, 0.2f, 0.4f},  // mb_lobe_L: darker pink
    {0.8f, 0.2f, 0.4f},  // mb_lobe_R: darker pink
    {0.3f, 0.9f, 0.4f},  // antennal_lobe_L: green
    {0.3f, 0.9f, 0.4f},  // antennal_lobe_R: green
    {1.0f, 0.8f, 0.2f},  // central_complex: gold
    {0.6f, 0.3f, 0.9f},  // lateral_horn_L: purple
    {0.6f, 0.3f, 0.9f},  // lateral_horn_R: purple
    {0.4f, 0.8f, 0.8f},  // sez: teal
  };
  int idx = (region >= 0 && region < 13) ? region : 0;
  r = colors[idx][0];
  g = colors[idx][1];
  b = colors[idx][2];
}

// ---------------------------------------------------------------------------
// Fly Sandbox: 2D virtual arena with odor plumes, sensory–motor coupling
// ---------------------------------------------------------------------------

// Virtual odor source in the arena
struct OdorSource {
  float x, y;      // position in arena coords [0..kArenaSize]
  float strength;  // peak concentration [0..1]
  float r, g, b;   // display color
};

// Fly agent state
struct FlyState {
  float x = 200.0f, y = 200.0f;  // arena position
  float heading = 0.0f;           // radians (0 = +x / east)
  float odor_L = 0.0f, odor_R = 0.0f;  // odor at each antenna [0..1]
  bool flying = false;            // true = airborne (Shift held)
  bool running = false;           // true = sprint (Space held)
  float wing_phase = 0.0f;       // wing animation phase [0..2pi]

  // EMA motor rates (Hz) read from brain regions
  float rate_lh_L = 0.0f;  // lateral horn L → steer right
  float rate_lh_R = 0.0f;  // lateral horn R → steer left
  float rate_sez   = 0.0f; // SEZ → forward speed

  // Trail ring buffer
  static constexpr int kTrailLen = 400;
  float trail_x[kTrailLen] = {};
  float trail_y[kTrailLen] = {};
  int   trail_head  = 0;
  int   trail_count = 0;

  void Reset(float cx, float cy) {
    x = cx; y = cy; heading = 0.0f;
    odor_L = odor_R = 0.0f;
    flying = false; running = false; wing_phase = 0.0f;
    rate_lh_L = rate_lh_R = rate_sez = 0.0f;
    trail_head = trail_count = 0;
  }

  void PushTrail() {
    trail_x[trail_head] = x;
    trail_y[trail_head] = y;
    trail_head = (trail_head + 1) % kTrailLen;
    trail_count = std::min(trail_count + 1, kTrailLen);
  }
};

// Generate brain point cloud from SDF
static std::vector<PointVertex> GenerateBrainPoints(
    const BrainSDF& sdf, float spacing) {
  std::vector<PointVertex> points;
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> jitter(-spacing * 0.4f, spacing * 0.4f);

  for (float z = 0; z < 200; z += spacing) {
    for (float y = 0; y < 300; y += spacing) {
      for (float x = 0; x < 500; x += spacing) {
        float jx = x + jitter(rng);
        float jy = y + jitter(rng);
        float jz = z + jitter(rng);
        float d = sdf.Evaluate(jx, jy, jz);
        if (d < -spacing * 0.3f) {
          int region = sdf.NearestRegion(jx, jy, jz);
          PointVertex pv;
          pv.x = jx; pv.y = jy; pv.z = jz;
          pv.region = region;
          RegionColor(region, pv.r, pv.g, pv.b);
          points.push_back(pv);
        }
      }
    }
  }
  return points;
}

// Vertex shader
static const char* kVertSrc = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
layout(location = 2) in float aSize;
uniform mat4 uView;
uniform mat4 uProj;
out vec3 vColor;
void main() {
  gl_Position = uProj * uView * vec4(aPos, 1.0);
  gl_PointSize = aSize;
  vColor = aColor;
}
)";

// Fragment shader: round points
static const char* kFragSrc = R"(
#version 330 core
in vec3 vColor;
out vec4 FragColor;
void main() {
  vec2 coord = gl_PointCoord * 2.0 - 1.0;
  if (dot(coord, coord) > 1.0) discard;
  FragColor = vec4(vColor, 1.0);
}
)";

// Global state
static OrbitCamera g_camera;
static bool g_mouse_left_dragging = false;
static bool g_mouse_right_dragging = false;
static double g_last_mx = 0, g_last_my = 0;

// Body sim (MuJoCo fly).
static BodyViewport g_body;
static bool g_body_mouse_left = false;
static bool g_body_mouse_right = false;
static bool g_body_mouse_mid = false;
static double g_body_last_mx = 0, g_body_last_my = 0;

// Brain window callbacks
static void BrainScrollCallback(GLFWwindow*, double, double yoff) {
  g_camera.radius *= (yoff > 0) ? 0.9f : 1.1f;
  g_camera.radius = std::clamp(g_camera.radius, 50.0f, 2000.0f);
}

// Body window callbacks
static void BodyScrollCallback(GLFWwindow* w, double, double yoff) {
  if (ImGui::GetIO().WantCaptureMouse) return;
  if (g_body.ready) {
    int width, height;
    glfwGetWindowSize(w, &width, &height);
    g_body.ZoomCamera(yoff, height);
  }
}

static void KeyCallback(GLFWwindow* window, int key, int, int action, int) {
  if (action != GLFW_PRESS) return;
  if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(window, GLFW_TRUE);
  if (key == GLFW_KEY_R) g_camera.Reset();
}

// Load region sizes from TOML config. Reads keys under [regions].
static bool LoadRegionConfig(const char* path, float* sizes, int count,
                             const BrainSDF& sdf) {
  std::ifstream f(path);
  if (!f.is_open()) return false;
  std::string line;
  bool in_regions = false;
  while (std::getline(f, line)) {
    size_t start = line.find_first_not_of(" \t");
    if (start == std::string::npos || line[start] == '#') continue;
    // Section headers
    if (line[start] == '[') {
      in_regions = (line.find("[regions]") != std::string::npos);
      continue;
    }
    if (!in_regions) continue;
    // Parse "name = value"
    size_t eq = line.find('=');
    if (eq == std::string::npos) continue;
    std::string name = line.substr(start, line.find_last_not_of(" \t", eq - 1) - start + 1);
    float val = std::stof(line.substr(eq + 1));
    for (int i = 0; i < count && i < static_cast<int>(sdf.primitives.size()); ++i) {
      if (sdf.primitives[i].name == name) {
        sizes[i] = val;
        break;
      }
    }
  }
  return true;
}

// Save region sizes to TOML config.
static void SaveRegionConfig(const char* path, const float* sizes,
                             const BrainSDF& sdf) {
  std::ofstream f(path);
  if (!f.is_open()) return;
  f << "# FWMC Brain Viewer config\n";
  f << "# Set point_size to 0 to hide a region\n\n";
  f << "[regions]\n";
  for (size_t i = 0; i < sdf.primitives.size(); ++i) {
    f << sdf.primitives[i].name;
    int pad = 20 - static_cast<int>(sdf.primitives[i].name.size());
    for (int p = 0; p < pad; ++p) f << ' ';
    f << "= " << sizes[i] << "\n";
  }
}

int main() {
  FILE* dbgf = fopen("fwmc_viewer_debug.log", "w");
  auto dbg = [&](const char* msg) { if (dbgf) { fprintf(dbgf, "%s\n", msg); fflush(dbgf); } };
  dbg("=== fwmc-viewer starting ===");

  if (!glfwInit()) {
    dbg("FAIL: glfwInit");
    fprintf(stderr, "Failed to initialize GLFW\n");
    return 1;
  }
  dbg("glfwInit OK");

  // --- Body window (main): compat context for MuJoCo ---
  GLFWwindow* body_window = nullptr;
#ifdef FWMC_BODY_SIM
  dbg("creating body window...");
  glfwDefaultWindowHints();  // compat context for MuJoCo
  body_window = glfwCreateWindow(1600, 900, "flygame", nullptr, nullptr);
  if (body_window) {
    dbg("body window created");
    glfwMakeContextCurrent(body_window);
    gladLoadGL(glfwGetProcAddress);
    dbg("body GL loaded, calling g_body.Init()...");
    if (g_body.Init()) {
      dbg("body init OK");
      printf("[body] NeuroMechFly body sim ready\n");
    } else {
      dbg("body init FAILED");
    }
    glfwSwapInterval(1);
    glfwSetScrollCallback(body_window, BodyScrollCallback);
    glfwSetKeyCallback(body_window, KeyCallback);
  } else {
    dbg("body window creation FAILED");
  }
#endif

  // --- Brain GL context: hidden window, shared with body for texture access ---
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
  if (body_window) glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

  GLFWwindow* window = glfwCreateWindow(
      body_window ? 2 : 1600, body_window ? 2 : 900,
      "Brain", nullptr, body_window);
  if (!window) {
    fprintf(stderr, "Failed to create brain window\n");
    glfwTerminate();
    return 1;
  }
  glfwMakeContextCurrent(window);
  if (!body_window) {
    glfwSwapInterval(1);
    glfwSetScrollCallback(window, BrainScrollCallback);
    glfwSetKeyCallback(window, KeyCallback);
  }

  dbg("loading brain GL...");
  int gl_version = gladLoadGL(glfwGetProcAddress);
  if (!gl_version) {
    dbg("FAIL: brain GL load");
    fprintf(stderr, "Failed to load OpenGL\n");
    return 1;
  }
  dbg("brain GL OK");

  // Brain FBO: render point cloud to texture for embedding in ImGui.
  GLuint brain_fbo = 0, brain_tex = 0, brain_depth = 0;
  int brain_fbo_w = 512, brain_fbo_h = 384;
  auto CreateBrainFBO = [&](int w, int h) {
    if (brain_fbo) { glDeleteFramebuffers(1, &brain_fbo); glDeleteTextures(1, &brain_tex); glDeleteRenderbuffers(1, &brain_depth); }
    brain_fbo_w = w; brain_fbo_h = h;
    glGenFramebuffers(1, &brain_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, brain_fbo);
    glGenTextures(1, &brain_tex);
    glBindTexture(GL_TEXTURE_2D, brain_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, brain_tex, 0);
    glGenRenderbuffers(1, &brain_depth);
    glBindRenderbuffer(GL_RENDERBUFFER, brain_depth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, brain_depth);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
  };
  CreateBrainFBO(brain_fbo_w, brain_fbo_h);

  // ImGui: init on body window if available, else brain window.
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  GLFWwindow* imgui_win = body_window ? body_window : window;
  if (body_window) glfwMakeContextCurrent(body_window);
  ImGui_ImplGlfw_InitForOpenGL(imgui_win, true);
  ImGui_ImplOpenGL3_Init(body_window ? "#version 130" : "#version 330");
  ImGui::StyleColorsDark();
  if (body_window) glfwMakeContextCurrent(window);

  // Generate brain geometry
  printf("Generating brain SDF...\n");
  BrainSDF sdf;
  sdf.InitDrosophila();

  float point_spacing = 3.0f;
  printf("Sampling brain volume (spacing=%.1f um)...\n", point_spacing);
  auto points = GenerateBrainPoints(sdf, point_spacing);
  printf("Generated %zu brain points\n", points.size());

  // Store base colors separately so activity overlay doesn't accumulate
  struct BaseColor { float r, g, b; };
  std::vector<BaseColor> base_colors(points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    base_colors[i] = {points[i].r, points[i].g, points[i].b};
  }

  // Neural field
  VoxelGrid grid;
  grid.Init(50, 30, 20, 10.0f);
  size_t ch_sdf = grid.AddChannel("sdf");
  size_t ch_reg = grid.AddChannel("region");
  sdf.BakeToGrid(grid, ch_sdf, ch_reg);
  BrainSDF::DiffuseSmooth(grid, ch_sdf, 5);

  NeuralField field;
  field.ch_sdf = ch_sdf;
  field.Init(grid);

  // Spiking neural network (uses the actual FWMC simulation engine)
  printf("Building spiking network (%zu neurons)...\n", points.size());
  NeuronArray neurons;
  neurons.Resize(points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    neurons.x[i] = points[i].x;
    neurons.y[i] = points[i].y;
    neurons.z[i] = points[i].z;
    neurons.region[i] = static_cast<uint8_t>(points[i].region);
  }
  // Load FlyWire connectome spec (drosophila_full.brain = FlyWire-scale parameters)
  SpeciesDefaults species_defaults;  // populated after spec load
  BrainSpec fly_spec;
  {
    const char* spec_paths[] = {
      SOURCE_DIR "/../../examples/drosophila_full.brain",
      "examples/drosophila_full.brain",
      "../examples/drosophila_full.brain",
    };
    bool loaded = false;
    for (auto& sp : spec_paths) {
      auto res = BrainSpecLoader::Load(sp);
      if (res) {
        fly_spec = std::move(*res);
        printf("Loaded FlyWire spec: %s (%zu regions, %zu projections)\n",
               sp, fly_spec.regions.size(), fly_spec.projections.size());
        loaded = true;
        break;
      }
    }
    if (!loaded) {
      printf("Warning: drosophila_full.brain not found -- using generic connectivity\n");
      // Minimal fallback spec (generic Drosophila-like)
      RegionSpec al; al.name = "antennal_lobe"; al.n_neurons = 2600;
      al.internal_density = 0.08f; al.default_nt = kACh;
      RegionSpec mb; mb.name = "mushroom_body"; mb.n_neurons = 21500;
      mb.internal_density = 0.003f; mb.default_nt = kACh;
      RegionSpec lh; lh.name = "lateral_horn"; lh.n_neurons = 1500;
      lh.internal_density = 0.06f; lh.default_nt = kACh;
      RegionSpec cx; cx.name = "central_complex"; cx.n_neurons = 5000;
      cx.internal_density = 0.02f; cx.default_nt = kACh;
      RegionSpec ol; ol.name = "optic_lobe"; ol.n_neurons = 80000;
      ol.internal_density = 0.0005f; ol.default_nt = kACh;
      RegionSpec vnc; vnc.name = "ventral_nerve_cord"; vnc.n_neurons = 25000;
      vnc.internal_density = 0.001f; vnc.default_nt = kACh;
      RegionSpec pc; pc.name = "protocerebrum"; pc.n_neurons = 3655;
      pc.internal_density = 0.01f; pc.default_nt = kACh;
      fly_spec.regions = {al, mb, lh, cx, ol, vnc, pc};
      // AL → MB projection
      ProjectionSpec p0; p0.from_region = "antennal_lobe"; p0.to_region = "mushroom_body";
      p0.density = 0.002f; p0.nt_type = kACh; p0.weight_mean = 1.5f; p0.weight_std = 0.4f;
      // AL → LH
      ProjectionSpec p1; p1.from_region = "antennal_lobe"; p1.to_region = "lateral_horn";
      p1.density = 0.015f; p1.nt_type = kACh; p1.weight_mean = 1.3f; p1.weight_std = 0.3f;
      // LH → CX
      ProjectionSpec p2; p2.from_region = "lateral_horn"; p2.to_region = "central_complex";
      p2.density = 0.005f; p2.nt_type = kACh; p2.weight_mean = 1.0f; p2.weight_std = 0.3f;
      // CX → VNC
      ProjectionSpec p3; p3.from_region = "central_complex"; p3.to_region = "ventral_nerve_cord";
      p3.density = 0.002f; p3.nt_type = kACh; p3.weight_mean = 1.5f; p3.weight_std = 0.4f;
      fly_spec.projections = {p0, p1, p2, p3};
      fly_spec.global_weight_mean = 1.0f;
      fly_spec.global_weight_std  = 0.3f;
      fly_spec.seed = 42;
    }
  }
  species_defaults = fly_spec.GetDefaults();

  SynapseTable connectome = BuildFlyWireConnectome(points, points.size(), fly_spec);
  AssignFlyWireCellTypes(neurons, points, fly_spec);
  connectome.AssignPerNeuronTau(neurons);  // per-NT synaptic time constants

  // Per-cell-type Izhikevich parameters (heterogeneous dynamics)
  CellTypeManager cell_types;
  cell_types.AssignFromTypes(neurons);
  printf("[types] assigned per-neuron Izhikevich params for %zu neurons\n", neurons.n);

  float spike_time = 0.0f;

  // Per-SDF-region background current from spec (falls back to global).
  // SDF regions 0-12 map to spec regions via kSDFToSpecName.
  float sdf_background[13];
  {
    auto find_spec_bg = [&](const std::string& name) -> float {
      for (auto& reg : fly_spec.regions)
        if (reg.name == name && reg.background_mean >= 0.0f)
          return reg.background_mean;
      return fly_spec.background_current_mean;
    };
    for (int i = 0; i < 13; ++i)
      sdf_background[i] = find_spec_bg(kSDFToSpecName[i]);
  }

  // Central feature flags (all togglable from UI)
  SimFeatures features;

  // STDP (optional, toggled in UI) - species-appropriate params
  STDPParams stdp_params;
  stdp_params.a_plus = species_defaults.stdp_a_plus;
  stdp_params.a_minus = species_defaults.stdp_a_minus;
  stdp_params.tau_plus = species_defaults.stdp_tau_plus;
  stdp_params.tau_minus = species_defaults.stdp_tau_minus;

  // Synaptic scaling: homeostatic weight normalization (Turrigiano 2008)
  // Prevents STDP from driving all weights to bounds.
  SynapticScaling synaptic_scaling;
  synaptic_scaling.Init(neurons.n);
  int scaling_accumulator = 0;  // counts steps since last Apply()
  constexpr int kScalingInterval = 100;  // apply every 100ms of sim time

  // Gap junctions: electrical coupling in AL local interneurons
  // (Drosophila innexin-based; Yaksi & Wilson 2010)
  GapJunctionTable gap_junctions;
  gap_junctions.BuildFromRegion(neurons, 7, 0.02f, 0.5f, 42);  // AL_L LNs
  gap_junctions.BuildFromRegion(neurons, 8, 0.02f, 0.5f, 43);  // AL_R LNs
  printf("[gap] %zu gap junctions in antennal lobes\n", gap_junctions.Size());

  // Intrinsic homeostasis: slow bias adjustment to stabilize firing rates
  // (Marder & Goaillard 2006, Turrigiano 2008)
  IntrinsicHomeostasis homeostasis;
  // Target 2 Hz: conservative to prevent homeostasis from driving
  // runaway excitation. KCs should be nearly silent (~0.5 Hz),
  // PNs ~5 Hz, LNs ~4.6 Hz. 2 Hz is a safe global average.
  homeostasis.Init(neurons.n, 2.0f, 1.0f);
  homeostasis.SetTargetsFromTypes(neurons);  // per-cell-type firing rate targets
  homeostasis.learning_rate = 0.005f;  // gentler adjustment
  homeostasis.max_bias = 3.0f;         // tighter clamp (was 5)
  // features.homeostasis now in features.homeostasis

  // Spike-frequency adaptation (calcium-based sAHP)
  SpikeFrequencyAdaptation sfa;
  sfa.Init(neurons.n);

  // Temperature model: species-appropriate reference temperature.
  TemperatureModel temperature;
  temperature.enabled = true;
  temperature.reference_temp_c = species_defaults.ref_temperature_c;
  temperature.current_temp_c = species_defaults.ref_temperature_c + 3.0f;

  // Distance-dependent conduction delays.
  // Velocity from species defaults (um/ms = m/s * 1000).
  // Viewer uses 1ms timesteps; positions in micrometers.
  float conduction_v = species_defaults.conduction_velocity_m_s * 1000.0f;
  connectome.InitDistanceDelay(neurons, conduction_v, 1.0f);
  printf("[delay] ring_size=%zu slots for conduction delays\n",
         connectome.ring_size);

  // Short-term plasticity (Tsodyks-Markram) from species defaults.
  if (features.short_term_plasticity) {
    STPParams stp_params{species_defaults.stp_U_se,
                         species_defaults.stp_tau_d,
                         species_defaults.stp_tau_f};
    connectome.InitSTP(stp_params);
    printf("[stp] short-term plasticity enabled (U=%.2f, tau_d=%.0fms, tau_f=%.0fms)\n",
           stp_params.U_se, stp_params.tau_d, stp_params.tau_f);
  }

  // Inhibitory plasticity: Vogels iSTDP for E/I balance
  InhibitorySTDP istdp;

  // Neuromodulatory excitability effects
  NeuromodulatorEffects neuromod_effects;

  // NMDA receptor dynamics: slow excitatory conductance with Mg2+ block.
  // Provides coincidence detection for Hebbian associative learning.
  NMDAReceptor nmda;
  nmda.Init(neurons.n);
  printf("[nmda] NMDA receptors enabled (tau=%.0fms, Mg=%.1fmM, gain=%.2f)\n",
         nmda.tau_nmda_ms, nmda.mg_conc_mM, nmda.nmda_gain);

  // Structural plasticity: synapse pruning/sprouting (runs infrequently)
  StructuralPlasticity plasticity;
  std::mt19937 plasticity_rng(fly_spec.seed + 7);
  int plasticity_step = 0;  // track sim steps for update interval

  // Rate monitor: validates per-region firing rates against Drosophila literature.
  // SDF regions 0-12 map to spec region names via kSDFToSpecName.
  RateMonitor rate_monitor;
  std::vector<RegionRate> rate_snapshot;  // latest rates for UI display
  int rate_monitor_accum = 0;  // sim steps since last ComputeRates()
  constexpr int kRateMonitorInterval = 1000;  // compute rates every 1000 steps
  {
    std::vector<std::string> sdf_names(13);
    for (int i = 0; i < 13; ++i) sdf_names[i] = kSDFToSpecName[i];
    rate_monitor.Init(neurons, sdf_names, 1.0f);
    printf("[rate_monitor] %zu regions initialized with Drosophila reference rates\n",
           rate_monitor.regions.size());
  }

  // Motor output: map SEZ descending neurons + MB output to behavior.
  // Compute actual SEZ midline from neuron positions (not hardcoded 250).
  MotorOutput motor;
  float sez_midline = 250.0f;
  {
    float sez_min_x = 1e9f, sez_max_x = -1e9f;
    int sez_count = 0;
    for (size_t i = 0; i < neurons.n; ++i) {
      if (neurons.region[i] == 12) {
        sez_min_x = std::min(sez_min_x, neurons.x[i]);
        sez_max_x = std::max(sez_max_x, neurons.x[i]);
        sez_count++;
      }
    }
    sez_midline = (sez_min_x + sez_max_x) * 0.5f;
    printf("[motor] SEZ: %d neurons, x range [%.1f, %.1f], midline=%.1f\n",
           sez_count, sez_min_x, sez_max_x, sez_midline);
    motor.InitFromRegions(neurons, 12, 3, sez_midline);
    printf("[motor] L=%zu R=%zu descending neurons\n",
           motor.descending_left.size(), motor.descending_right.size());
  }

  // Proprioceptive feedback: maps body state to VNC sensory neuron currents.
  // VNC is region 5 in drosophila_full.brain. Closes the sensorimotor loop.
  ProprioMap proprio;
  ProprioConfig proprio_cfg;
  proprio.Init(neurons, 5, sez_midline);
  if (proprio.initialized) {
    printf("[proprio] VNC sensory map: %zu haltere neurons (L=%zu R=%zu)\n",
           proprio.haltere_left.size() + proprio.haltere_right.size(),
           proprio.haltere_left.size(), proprio.haltere_right.size());
  }

  // Central pattern generator: oscillatory drive to VNC for spontaneous locomotion.
  // The brain modulates CPG amplitude via descending neuron firing rate.
  CPGOscillator cpg;
  cpg.Init(neurons, 5, sez_midline);
  if (cpg.initialized) {
    printf("[cpg] VNC motor groups: A=%zu B=%zu neurons (tripod alternation)\n",
           cpg.group_a.size(), cpg.group_b.size());
  }

  // Spike raster: per-region spike count over last N frames
  constexpr int kRasterFrames = 200;
  std::vector<std::array<float, 13>> raster_history(kRasterFrames);
  int raster_head = 0;

  // Per-region spike rates (Hz)
  float region_rates[13] = {};
  int region_neuron_count[13] = {};
  for (size_t i = 0; i < neurons.n; ++i) {
    int r = neurons.region[i];
    if (r >= 0 && r < 13) region_neuron_count[r]++;
  }

  // Upload point cloud to GPU
  GLuint vao, vbo;
  glGenVertexArrays(1, &vao);
  glGenBuffers(1, &vbo);
  glBindVertexArray(vao);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER,
               static_cast<GLsizeiptr>(points.size() * sizeof(PointVertex)),
               points.data(), GL_DYNAMIC_DRAW);
  // Position
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(PointVertex),
                        (void*)offsetof(PointVertex, x));
  // Color
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(PointVertex),
                        (void*)offsetof(PointVertex, r));
  // Per-vertex point size
  glEnableVertexAttribArray(2);
  glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(PointVertex),
                        (void*)offsetof(PointVertex, size));
  glBindVertexArray(0);

  // Compile shaders
  GLuint program = LinkProgram(
      CompileShader(GL_VERTEX_SHADER, kVertSrc),
      CompileShader(GL_FRAGMENT_SHADER, kFragSrc));
  GLint loc_view = glGetUniformLocation(program, "uView");
  GLint loc_proj = glGetUniformLocation(program, "uProj");

  // Rendering state
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_PROGRAM_POINT_SIZE);

  // ---- Sim/Render state ----
  bool simulate = false;
  float sim_speed = 1.0f;
  bool realtime_mode = false;
  double brain_time_s = 0.0;
  double wall_time_base = glfwGetTime();
  bool show_activity = false;
  int sim_mode = 1;
  std::vector<float> spike_vis(points.size(), 0.0f);
  std::vector<float> stim_extra(neurons.n, 0.0f);
  int spike_count_display = 0;

  // Fly sandbox constants
  static constexpr float kArenaSize        = 400.0f;
  static constexpr float kOdorCurrentScale = 12.0f;  // [0..1] odor → i_ext (pA)
  static constexpr float kFlySpeedScale    = 0.25f;  // Hz → px/step
  static constexpr float kFlyTurnScale     = 0.004f; // Hz → rad/step
  static constexpr float kOdorSigma2       = 8000.0f;// odor plume falloff (px²)

  // Fly sandbox state
  FlyState fly;
  fly.Reset(kArenaSize * 0.5f, kArenaSize * 0.5f);
  bool manual_fly = false;  // WASD control mode

  // Spike union buffer for batched propagation (allocated once)
  std::vector<uint8_t> spike_union(neurons.n, 0);

  // TCP server for nmfly body connection
  ViewerTcpServer tcp_server;
  bool tcp_enabled = false;


  std::vector<OdorSource> odor_sources = {
    {320.0f, 190.0f, 1.0f, 0.2f, 0.85f, 0.3f},  // green (attractive)
    { 80.0f, 290.0f, 0.8f, 0.9f, 0.30f, 0.1f},  // red   (second source)
  };
  float region_size[14] = {4,4,4,4,4,4,4,4,4,4,4,4,4,4};
  bool region_dirty = false;
  // Config path: next to the source file for easy editing
  const char* config_candidates[] = {
    SOURCE_DIR "/regions.toml",
    "regions.toml",
    "../viewer/regions.toml",
  };
  const char* config_path = config_candidates[0];
  for (auto& cp : config_candidates) {
    if (std::filesystem::exists(cp)) { config_path = cp; break; }
  }
  LoadRegionConfig(config_path, region_size, 14, sdf);

  // Set initial per-vertex sizes
  for (auto& pv : points) {
    pv.size = region_size[pv.region];
  }
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferSubData(GL_ARRAY_BUFFER, 0,
                  static_cast<GLsizeiptr>(points.size() * sizeof(PointVertex)),
                  points.data());

  // FPS tracking
  double fps_time = glfwGetTime();
  int fps_count = 0;
  float fps_display = 0.0f;

  // Auto-start: fly sandbox with WASD control, TCP off by default
  sim_mode = 2;
  manual_fly = true;
  simulate = true;
  printf("Fly sandbox active. WASD to control.\n");

  // Main loop
  dbg("entering main loop");
  if (dbgf) { fclose(dbgf); dbgf = nullptr; }  // flush and close debug log
  GLFWwindow* main_window = body_window ? body_window : window;
  while (!glfwWindowShouldClose(main_window)) {
    glfwPollEvents();

    // FPS
    fps_count++;
    double now = glfwGetTime();
    if (now - fps_time >= 0.5) {
      fps_display = static_cast<float>(fps_count / (now - fps_time));
      fps_count = 0;
      fps_time = now;
    }

    // Mouse input: brain camera controls via ImGui image interaction (see below)
    // or direct GLFW when no body window.
    if (!body_window && !ImGui::GetIO().WantCaptureMouse) {
      double mx, my;
      glfwGetCursorPos(window, &mx, &my);

      bool left = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
      bool right = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
      bool mid = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;

      {
        // Brain camera controls.
        if (left && !g_mouse_left_dragging) {
          g_mouse_left_dragging = true;
          g_last_mx = mx; g_last_my = my;
        }
        if (!left) g_mouse_left_dragging = false;

        if ((right || mid) && !g_mouse_right_dragging) {
          g_mouse_right_dragging = true;
          g_last_mx = mx; g_last_my = my;
        }
        if (!right && !mid) g_mouse_right_dragging = false;

        if (g_mouse_left_dragging) {
          float dx = static_cast<float>(mx - g_last_mx);
          float dy = static_cast<float>(my - g_last_my);
          g_camera.azimuth += dx * 0.005f;
          g_camera.elevation += dy * 0.005f;
          g_camera.elevation = std::clamp(g_camera.elevation, -1.5f, 1.5f);
          g_last_mx = mx; g_last_my = my;
        }
        if (g_mouse_right_dragging) {
          float dx = static_cast<float>(mx - g_last_mx);
          float dy = static_cast<float>(my - g_last_my);
          float pan_speed = g_camera.radius * 0.002f;
          g_camera.target_x -= dx * pan_speed * std::cos(g_camera.azimuth);
          g_camera.target_z += dx * pan_speed * std::sin(g_camera.azimuth);
          g_camera.target_y += dy * pan_speed;
          g_last_mx = mx; g_last_my = my;
        }

      }
    }

    // Simulation
    if (simulate) {
      // Batched spike propagation: run N cheap ODE substeps per one
      // expensive CSR synapse walk. Preserves 1ms spike timing while
      // amortizing the synapse traversal cost. The added latency (N ms)
      // is within biological synaptic delay range.
      // One propagation per frame: all substeps feed a single CSR walk.
      // ~33ms latency at 30fps, within cortical processing delay range.
      int total_ms;  // total brain milliseconds to simulate this frame
      if (realtime_mode) {
        double wall_now = glfwGetTime();
        double wall_elapsed = (wall_now - wall_time_base) * static_cast<double>(sim_speed);
        double deficit_ms = wall_elapsed * 1000.0 - brain_time_s * 1000.0;
        total_ms = std::clamp(static_cast<int>(deficit_ms), 1, 100);
        brain_time_s += total_ms * 0.001;
      } else {
        total_ms = static_cast<int>(sim_speed);
      }

      if (sim_mode == 0) {
        // Wave mode (reaction-diffusion neural field)
        for (int s = 0; s < total_ms; ++s)
          field.Step(grid, 0.5f);

        if (show_activity) {
          for (size_t i = 0; i < points.size(); ++i) {
            auto& pv = points[i];
            float activity = grid.Sample(field.ch_e, pv.x, pv.y, pv.z);
            float a = std::clamp(activity * 3.0f, 0.0f, 1.0f);
            float hot_r = std::clamp(a * 3.0f, 0.0f, 1.0f);
            float hot_g = std::clamp(a * 3.0f - 1.0f, 0.0f, 1.0f);
            float hot_b = std::clamp(a * 3.0f - 2.0f, 0.0f, 1.0f);
            pv.r = base_colors[i].r * (1.0f - a) + hot_r * a;
            pv.g = base_colors[i].g * (1.0f - a) + hot_g * a;
            pv.b = base_colors[i].b * (1.0f - a) + hot_b * a;
          }
        }
      } else {
        // Modes 1 (manual spiking) and 2 (fly sandbox)

        // --- Fly sandbox: inject sensory inputs before stepping ---
        if (sim_mode == 2) {
          if (manual_fly) {
            // WASD = walk, Space = fly
            // Use GetAsyncKeyState (Win32) so keys work even when
            // another window (flygym) has focus. No keyboard hook needed.
            auto key_down = [](int vk) -> bool {
#ifdef _WIN32
              return (GetAsyncKeyState(vk) & 0x8000) != 0;
#else
              (void)vk; return false;
#endif
            };

            // Smooth drive signals: ramp up on press, decay on release.
            // Models tonic descending neuron activation (like dopaminergic
            // locomotor drive). Tau ~150ms rise, ~300ms decay.
            static float smooth_fwd = 0.0f;
            static float smooth_turn = 0.0f;
            static bool smooth_flying = false;
            static bool smooth_running = false;
            constexpr float kRiseRate = 0.45f;   // per sim step: snappy saccade-like onset
            constexpr float kDecayRate = 0.25f;  // per sim step: quick stop
            constexpr float kBrakeRate = 0.30f;  // S key: active braking

            float target_fwd = key_down('W') ? 1.0f : 0.0f;
            if (key_down('S')) { smooth_fwd -= kBrakeRate; target_fwd = 0.0f; }
            float target_turn = 0.0f;
            if (key_down('A')) target_turn -= 1.0f;
            if (key_down('D')) target_turn += 1.0f;
            smooth_flying = key_down(VK_SHIFT);
            smooth_running = key_down(VK_SPACE);

            // Exponential approach to target
            float fwd_rate = (target_fwd > smooth_fwd) ? kRiseRate : kDecayRate;
            smooth_fwd += (target_fwd - smooth_fwd) * fwd_rate;
            smooth_fwd = std::clamp(smooth_fwd, 0.0f, 1.0f);
            float turn_rate = (std::abs(target_turn) > std::abs(smooth_turn))
                              ? kRiseRate : kDecayRate;
            smooth_turn += (target_turn - smooth_turn) * turn_rate;
            smooth_turn = std::clamp(smooth_turn, -1.0f, 1.0f);
            // Kill tiny residual
            if (std::abs(smooth_fwd) < 0.01f) smooth_fwd = 0.0f;
            if (std::abs(smooth_turn) < 0.01f) smooth_turn = 0.0f;

            float drive_fwd = smooth_fwd;
            float drive_turn = smooth_turn;
            fly.flying = smooth_flying;
            fly.running = smooth_running && !smooth_flying;

            // Three locomotion tiers:
            //   Walk (default): 1x speed, 12 pA base current
            //   Run  (Space):   2x speed, 15 pA base current
            //   Fly  (Shift):   3x speed, 18 pA base current, wing oscillation
            float speed_mult = fly.flying ? 3.0f : (fly.running ? 2.0f : 1.0f);
            float base_current = fly.flying ? 18.0f : (fly.running ? 15.0f : 12.0f);

            // SEZ split L/R for turning.
            // W = both sides equal. A/D alone = turn in place (one side
            // forward, other side zero). W+A/D = walk + arc turn.
            float sez_fwd = drive_fwd * base_current * speed_mult;
            float turn_strength = std::abs(drive_turn) * base_current;
            // Turn in place: drive one side, suppress the other
            float sez_L = sez_fwd + (drive_turn > 0 ? turn_strength : -turn_strength * 0.8f);
            float sez_R = sez_fwd + (drive_turn < 0 ? turn_strength : -turn_strength * 0.8f);
            float lh_L_current = drive_turn > 0 ? drive_turn * base_current : 0.0f;
            float lh_R_current = drive_turn < 0 ? -drive_turn * base_current : 0.0f;

#ifdef FWMC_BODY_SIM
            // Proprioceptive feedback: haltere-like yaw rate sensing.
            // If the fly is drifting left (positive yaw), inhibit left SEZ
            // and excite right SEZ to correct. This is what real halteres do.
            float yaw_rate = g_body.ready ? g_body.YawRate() : 0.0f;
            constexpr float kHaltereGain = 0.5f;  // feedback strength
            float haltere_L = -yaw_rate * kHaltereGain;  // left turn → inhibit left
            float haltere_R =  yaw_rate * kHaltereGain;  // left turn → excite right
            sez_L += haltere_L;
            sez_R += haltere_R;
#endif

            // Clamp: negative current doesn't work well on neurons
            sez_L = std::max(0.0f, sez_L);
            sez_R = std::max(0.0f, sez_R);

            // Midline for L/R split (computed from actual SEZ neuron positions)
            float midline_x = sez_midline;

            // Tonic inhibition on motor regions (SEZ, lateral horn):
            // real descending neurons are held silent by GABAergic tone.
            // WASD-injected excitation must overcome this to produce movement.
            constexpr float kTonicInhib = -3.0f;

            for (size_t i = 0; i < neurons.n; ++i) {
              uint8_t r = neurons.region[i];
              if (r == 12) {
                // SEZ: tonic inhibition + WASD excitation
                float drive = (neurons.x[i] < midline_x) ? sez_L : sez_R;
                neurons.i_ext[i] = kTonicInhib + drive;
              }
              else if (r == 10) neurons.i_ext[i] = kTonicInhib + lh_L_current;
              else if (r == 11) neurons.i_ext[i] = kTonicInhib + lh_R_current;
              else if (fly.flying) {
                // Flight engages central complex + optic lobes + mushroom body
                float wing_drive = (r <= 2 || r == 5 || r == 6) ? 6.0f : 0.0f;
                neurons.i_ext[i] = wing_drive;
              }
              else {
                // Per-region background drive from spec
                // (tonic synaptic bombardment, keeps neurons near threshold)
                neurons.i_ext[i] = sdf_background[r];
              }
              // Add persistent button stimulation on top
              neurons.i_ext[i] += stim_extra[i];
            }

            // Proprioceptive feedback: inject body state into VNC neurons.
            // Reads joint angles, contacts, and velocity from MuJoCo body.
            // Also provides ground contacts to entrain CPG rhythm.
#ifdef FWMC_BODY_SIM
            if (g_body.ready && proprio.initialized) {
              ProprioState ps = ReadProprioFromMuJoCo(g_body.mj_model,
                                                      g_body.mj_data);
              proprio.Inject(neurons, ps, proprio_cfg);
              cpg.StepWithFeedback(neurons, 1.0f, drive_fwd, ps.contacts);
            } else {
              cpg.Step(neurons, 1.0f, drive_fwd);
            }
#else
            cpg.Step(neurons, 1.0f, drive_fwd);
#endif
          } else {
            // Autonomous: odor-driven input via antennae
            float ant_L_x = fly.x + 20.0f * std::cos(fly.heading + 0.5f);
            float ant_L_y = fly.y + 20.0f * std::sin(fly.heading + 0.5f);
            float ant_R_x = fly.x + 20.0f * std::cos(fly.heading - 0.5f);
            float ant_R_y = fly.y + 20.0f * std::sin(fly.heading - 0.5f);

            float odorL = 0.0f, odorR = 0.0f;
            for (auto& src : odor_sources) {
              float dL2 = (ant_L_x-src.x)*(ant_L_x-src.x)+(ant_L_y-src.y)*(ant_L_y-src.y);
              float dR2 = (ant_R_x-src.x)*(ant_R_x-src.x)+(ant_R_y-src.y)*(ant_R_y-src.y);
              odorL += src.strength * std::exp(-dL2 / kOdorSigma2);
              odorR += src.strength * std::exp(-dR2 / kOdorSigma2);
            }
            fly.odor_L = std::min(1.0f, odorL);
            fly.odor_R = std::min(1.0f, odorR);

            for (size_t i = 0; i < neurons.n; ++i) {
              if (neurons.region[i] == 7)       // antennal_lobe_L
                neurons.i_ext[i] = fly.odor_L * kOdorCurrentScale;
              else if (neurons.region[i] == 8)  // antennal_lobe_R
                neurons.i_ext[i] = fly.odor_R * kOdorCurrentScale;
            }
          }
        }

        // --- Sim/render decoupling ---
        // The expensive work (ODE substeps + CSR synapse walk) runs on a
        // background thread via std::async. The render thread checks if
        // the result is ready (non-blocking) and consumes it. If the sim
        // is still running, the render just reuses the previous frame's
        // visual data, keeping FPS stable regardless of sim cost.
        constexpr float kWeightScale = 0.3f;
        static std::future<void> sim_future;
        static bool sim_pending = false;
        // These persist across frames (file-scope statics can't be
        // captured by lambda, so we use a local struct instead).
        struct SimState {
          float spike_frac = 0.0f;
          float governor = 1.0f;
          float smooth_fwd_vel = 0.0f;
          float smooth_ang_vel = 0.0f;
        };
        static SimState ss;

        // Check if background sim finished (non-blocking)
        bool sim_just_finished = false;
        if (sim_pending) {
          if (sim_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            sim_future.get();
            sim_pending = false;
            sim_just_finished = true;
          }
        }

        // --- Process results from completed sim step ---
        if (sim_just_finished) {
          if (features.stdp) {
            STDPUpdate(connectome, neurons, spike_time, stdp_params);

            // Inhibitory STDP: Vogels rule for E/I balance
            if (features.inhibitory_plasticity) {
              if (!istdp.IsInitialized())
                istdp.Init(connectome.post.size(), neurons.n);
              InhibitorySTDPUpdate(connectome, neurons,
                                    static_cast<float>(total_ms), istdp);
            }

            // Three-factor learning: convert eligibility traces to weight changes
            if (stdp_params.use_eligibility_traces && stdp_params.dopamine_gated) {
              EligibilityTraceUpdate(connectome, neurons,
                                      static_cast<float>(total_ms), stdp_params);
            }

            // Synaptic scaling: periodically normalize weights to prevent
            // STDP from driving all weights to their bounds
            synaptic_scaling.AccumulateSpikes(neurons, static_cast<float>(total_ms));
            scaling_accumulator += total_ms;
            if (scaling_accumulator >= kScalingInterval) {
              synaptic_scaling.Apply(connectome, stdp_params);
              scaling_accumulator = 0;
            }
          }

          // Structural plasticity: prune weak synapses, sprout new ones
          // between co-active neurons. Runs infrequently (every ~10s of sim
          // time) so the CSR rebuild cost is amortized.
          if (features.structural_plasticity) {
            plasticity_step += total_ms;
            plasticity.Update(connectome, neurons, plasticity_step, plasticity_rng);
          }

          // Motor output
          motor.Update(neurons, 1.0f);
          if (g_body.ready) {
            auto cmd = motor.Command();
            constexpr float kBodyTau = 15.0f;
            float body_alpha = 1.0f - std::exp(-static_cast<float>(total_ms) / kBodyTau);
            ss.smooth_fwd_vel += body_alpha * (cmd.forward_velocity - ss.smooth_fwd_vel);
            ss.smooth_ang_vel += body_alpha * (cmd.angular_velocity - ss.smooth_ang_vel);
            if (cmd.freeze > 0.5f) { ss.smooth_fwd_vel = 0.0f; ss.smooth_ang_vel = 0.0f; }
            double magnitude = std::clamp(
                static_cast<double>(ss.smooth_fwd_vel) / 10.0, 0.0, 1.5);
            double dturn_val = std::clamp(
                static_cast<double>(ss.smooth_ang_vel) / 0.8, -1.0, 1.0);
#ifdef FWMC_BODY_SIM
            g_body.cpg.SetDrive(magnitude, dturn_val);
#endif
          }
          spike_count_display = neurons.CountSpikes();

          // Rate monitor: compute rates periodically
          if (features.rate_monitor && rate_monitor_accum >= kRateMonitorInterval) {
            rate_snapshot = rate_monitor.ComputeRates();
            rate_monitor_accum = 0;
          }

          // Body physics
#ifdef FWMC_BODY_SIM
          if (g_body.ready) g_body.Step();
#endif

          // Fly sandbox kinematics
          if (sim_mode == 2) {
            auto inst_rate = [&](int reg) -> float {
              int count = 0, total_n = 0;
              for (size_t i = 0; i < neurons.n; ++i) {
                if (neurons.region[i] == static_cast<uint8_t>(reg)) {
                  count += neurons.spiked[i];
                  total_n++;
                }
              }
              return total_n > 0 ? static_cast<float>(count) / total_n * 1000.0f : 0.0f;
            };

            float alpha = 0.07f;
            fly.rate_lh_L += alpha * (inst_rate(10) - fly.rate_lh_L);
            fly.rate_lh_R += alpha * (inst_rate(11) - fly.rate_lh_R);
            fly.rate_sez   += alpha * (inst_rate(12) - fly.rate_sez);

            float loco_mult = fly.flying ? 3.0f : (fly.running ? 2.0f : 1.0f);
            float fwd  = std::clamp(fly.rate_sez * kFlySpeedScale * loco_mult, -2.0f, 24.0f);
            float turn_val = (fly.rate_lh_L - fly.rate_lh_R) * kFlyTurnScale * (fly.flying ? 1.5f : 1.0f);
            if (fly.flying) fly.wing_phase += 0.5f;

            fly.heading += turn_val;
            fly.x += std::cos(fly.heading) * fwd;
            fly.y += std::sin(fly.heading) * fwd;

            const float margin = 15.0f;
            if (fly.x < margin)             { fly.x = margin;             fly.heading = 3.14159f - fly.heading; }
            if (fly.x > kArenaSize - margin){ fly.x = kArenaSize - margin; fly.heading = 3.14159f - fly.heading; }
            if (fly.y < margin)             { fly.y = margin;             fly.heading = -fly.heading; }
            if (fly.y > kArenaSize - margin){ fly.y = kArenaSize - margin; fly.heading = -fly.heading; }
            fly.PushTrail();
          }

          // TCP bridge
          if (tcp_enabled) {
            auto cmd = motor.Command();
            if (sim_mode == 2) cmd.freeze = 0.0f;
            tcp_server.BufferMotorCommand(cmd);
            tcp_server.Poll();
            if (tcp_server.HasClient() && tcp_server.HasNewBioReadings()) {
              tcp_server.SendMotorBatch();
              tcp_server.ClearBioReadings();
            }
          }

          // Record per-region spike counts for raster
          std::array<float, 13> frame_counts = {};
          for (size_t i = 0; i < neurons.n; ++i) {
            int r = neurons.region[i];
            if (r >= 0 && r < 13) frame_counts[r] += neurons.spiked[i];
          }
          for (int r = 0; r < 13; ++r) {
            if (region_neuron_count[r] > 0)
              frame_counts[r] /= static_cast<float>(region_neuron_count[r]);
            region_rates[r] = frame_counts[r] * 1000.0f;
          }
          raster_history[raster_head] = frame_counts;
          raster_head = (raster_head + 1) % kRasterFrames;

          // Visual: spike flash decays over frames
          for (size_t i = 0; i < points.size(); ++i) {
            if (neurons.spiked[i]) spike_vis[i] = 1.0f;
            else spike_vis[i] *= 0.85f;

            float a = spike_vis[i];
            points[i].r = base_colors[i].r * (1.0f - a) + a;
            points[i].g = base_colors[i].g * (1.0f - a) + a;
            points[i].b = base_colors[i].b * (1.0f - a) + a;
            float base_sz = region_size[points[i].region];
            points[i].size = base_sz + a * 3.0f;
          }
        }

        // --- Launch new sim step if none pending ---
        if (!sim_pending) {
          // Prepare neuron input (main thread only, sim not running)
          float stim_decay_total = std::pow(0.997f, static_cast<float>(total_ms));
          for (size_t i = 0; i < neurons.n; ++i)
            stim_extra[i] *= stim_decay_total;

          // Capture values for the async lambda
          bool homeo_on = features.homeostasis;
          bool gap_on = features.gap_junctions;
          int steps = total_ms;

          bool sfa_on = features.sfa;
          bool neuromod_on = features.neuromodulation;
          bool delays_on = features.conduction_delays;
          bool stp_on = features.short_term_plasticity && connectome.HasSTP();
          bool neuromod_effects_on = features.neuromodulator_effects && features.neuromodulation;
          bool rate_mon_on = features.rate_monitor;
          bool per_step = features.per_step_propagation;
          bool nmda_on = features.nmda;
          float syn_tau = temperature.ScaledTauSyn(species_defaults.tau_syn_excitatory);

          SimState* pss = &ss;
          sim_future = std::async(std::launch::async,
            [&neurons, &spike_union, &gap_junctions, &cell_types,
             &homeostasis, &sfa, &connectome, &spike_time,
             &neuromod_effects, &rate_monitor, &nmda,
             pss, homeo_on, gap_on, sfa_on, neuromod_on, delays_on,
             stp_on, neuromod_effects_on, rate_mon_on, per_step,
             nmda_on, syn_tau, steps, kWeightScale,
             &rate_monitor_accum]() {
              std::memset(spike_union.data(), 0, neurons.n);
              const int nn = static_cast<int>(neurons.n);

              // Per-step governor: running spike count for anti-runaway scaling
              int running_spikes = 0;

              for (int s = 0; s < steps; ++s) {
                if (delays_on) connectome.DeliverDelayed(neurons.i_syn.data());
                neurons.DecaySynapticInput(1.0f, syn_tau);
                if (gap_on) gap_junctions.PropagateGapCurrents(neurons);
                if (stp_on) connectome.RecoverSTP(1.0f);
                if (nmda_on) nmda.Step(neurons, 1.0f);
                IzhikevichStepHeterogeneousFast(neurons, 1.0f, spike_time, cell_types);
                if (sfa_on) sfa.Update(neurons, 1.0f);
                if (neuromod_on) NeuromodulatorUpdate(neurons, connectome, 1.0f);
                if (neuromod_effects_on) neuromod_effects.Apply(neurons);
                if (rate_mon_on) {
                  rate_monitor.RecordStep(neurons);
                  ++rate_monitor_accum;
                }
                if (homeo_on) {
                  homeostasis.RecordSpikes(neurons);
                  homeostasis.MaybeApply(neurons);
                }

                // Per-substep spike propagation: each step's spikes drive
                // synaptic current immediately, enabling within-frame recurrent
                // dynamics. Matches fwmc.cc per-step behavior.
                if (per_step) {
                  int step_spikes = 0;
                  for (int i = 0; i < nn; ++i)
                    step_spikes += neurons.spiked[i];
                  running_spikes += step_spikes;

                  float step_frac = static_cast<float>(step_spikes) /
                                    static_cast<float>(nn);
                  float step_gov = (step_frac > 0.15f)
                      ? 0.15f / step_frac : 1.0f;

                  connectome.PropagateSpikes(neurons.spiked.data(),
                                             neurons.i_syn.data(),
                                             kWeightScale * step_gov);
                  if (nmda_on)
                    nmda.AccumulateFromSpikes(connectome, neurons.spiked.data(),
                                              kWeightScale * step_gov);
                }

                for (int i = 0; i < nn; ++i)
                  spike_union[i] |= neurons.spiked[i];
                if (delays_on) connectome.AdvanceDelayRing();
                spike_time += 1.0f;
              }

              // Frame-level governor (for display and batch fallback)
              int union_spike_count = 0;
              for (size_t i = 0; i < neurons.n; ++i)
                union_spike_count += spike_union[i];
              pss->spike_frac = static_cast<float>(union_spike_count) /
                                static_cast<float>(neurons.n);
              pss->governor = (pss->spike_frac > 0.15f)
                  ? 0.15f / pss->spike_frac : 1.0f;

              // Batch propagation fallback (when per-step is off)
              if (!per_step) {
                connectome.PropagateSpikes(spike_union.data(),
                                           neurons.i_syn.data(),
                                           kWeightScale * pss->governor);
                if (nmda_on)
                  nmda.AccumulateFromSpikes(connectome, spike_union.data(),
                                            kWeightScale * pss->governor);
              }
          });
          sim_pending = true;
        }
      }

      glBindBuffer(GL_ARRAY_BUFFER, vbo);
      glBufferSubData(GL_ARRAY_BUFFER, 0,
                      static_cast<GLsizeiptr>(points.size() * sizeof(PointVertex)),
                      points.data());
    }

    // ImGui frame (on body window if available, else brain window)
    if (body_window) glfwMakeContextCurrent(body_window);
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    if (body_window) glfwMakeContextCurrent(window);

    // Show panel on the body window (or brain if no body).
    GLFWwindow* panel_win = body_window ? body_window : window;
    int win_w_now, win_h_now;
    glfwGetWindowSize(panel_win, &win_w_now, &win_h_now);
    const bool show_panel = (win_w_now > 500 && win_h_now > 400);

    if (show_panel) {
    // Unified panel (Unity Inspector style)
    char title_buf[64];
    snprintf(title_buf, sizeof(title_buf), "flygame | %.0f FPS###main",
             fps_display);
    float panel_x = 10.0f;
    float panel_h = static_cast<float>(win_h_now) - 20.0f;
    ImGui::SetNextWindowPos(ImVec2(panel_x, 10), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(300, panel_h), ImGuiCond_Always);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(6, 4));
    ImGui::Begin(title_buf);

    // -- Simulation (collapsing header) --
    if (ImGui::CollapsingHeader("Simulation", ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::PushID("sim");
      ImGui::Checkbox("Simulate", &simulate);
      ImGui::SameLine();
      ImGui::SetNextItemWidth(-1);
      ImGui::SliderFloat("##speed", &sim_speed, 1.0f, 20.0f, "Speed %.0fx");
      if (ImGui::Checkbox("Real-time", &realtime_mode)) {
        // Reset time base when toggling
        wall_time_base = glfwGetTime();
        brain_time_s = 0.0;
      }

      ImGui::Text("Spikes: %d / %zu", spike_count_display, points.size());
      ImGui::Text("Synapses: %zu | t=%.0f ms", connectome.Size(), spike_time);

      // Stimulation buttons
      float bw = (ImGui::GetContentRegionAvail().x - 8) / 3.0f;
      {
        auto stim_region = [&](int region, float current) {
          for (size_t i = 0; i < neurons.n; ++i) {
            if (neurons.region[i] == region)
              stim_extra[i] = current;
          }
        };
        if (ImGui::Button("Antennal", ImVec2(bw, 0))) {
          stim_region(7, 15.0f);
          stim_region(8, 15.0f);
        }
        ImGui::SameLine();
        if (ImGui::Button("Optic", ImVec2(bw, 0))) {
          stim_region(1, 15.0f);
          stim_region(2, 15.0f);
        }
        ImGui::SameLine();
        if (ImGui::Button("All", ImVec2(bw, 0))) {
          for (size_t i = 0; i < neurons.n; ++i)
            stim_extra[i] = 12.0f;
        }
        // Second row of stim buttons
        if (ImGui::Button("Mushroom", ImVec2(bw, 0))) {
          stim_region(3, 15.0f);
          stim_region(4, 15.0f);
        }
        ImGui::SameLine();
        if (ImGui::Button("CX", ImVec2(bw, 0))) {
          stim_region(9, 15.0f);
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset", ImVec2(bw, 0))) {
          std::fill(stim_extra.begin(), stim_extra.end(), 0.0f);
          neurons.ClearExternalInput();
          for (size_t i = 0; i < neurons.n; ++i) {
            neurons.v[i] = -65.0f;
            neurons.u[i] = -13.0f;
            neurons.spiked[i] = 0;
            neurons.last_spike_time[i] = -1000.0f;
          }
          std::fill(spike_vis.begin(), spike_vis.end(), 0.0f);
          spike_time = 0.0f;
        }
      }
      ImGui::PopID();
    }

    // -- Activity Monitor (spike raster + rates + motor) --
    if ((sim_mode == 1 || sim_mode == 2) &&
        ImGui::CollapsingHeader("Activity Monitor", ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::PushID("activity");

      // Per-region firing rates (compact table)
      static const char* region_labels[] = {
        "Proto", "OL-L", "OL-R", "Cal-L", "Cal-R",
        "Lob-L", "Lob-R", "AL-L", "AL-R", "CX",
        "LH-L", "LH-R", "SEZ"
      };
      if (ImGui::TreeNodeEx("Firing Rates",
              ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_SpanAvailWidth)) {
        for (int r = 0; r < 13; ++r) {
          if (region_neuron_count[r] == 0) continue;
          float rate = region_rates[r];
          // Color: green if plausible (1-30 Hz), yellow if high, red if extreme
          ImVec4 col = (rate < 1.0f) ? ImVec4(0.5f, 0.5f, 0.5f, 1.0f)
                     : (rate < 30.0f) ? ImVec4(0.3f, 1.0f, 0.3f, 1.0f)
                     : (rate < 60.0f) ? ImVec4(1.0f, 1.0f, 0.3f, 1.0f)
                     :                  ImVec4(1.0f, 0.3f, 0.3f, 1.0f);
          ImGui::TextColored(col, "%-5s %6.1f Hz", region_labels[r], rate);
        }
        ImGui::TreePop();
      }

      // Spike raster: stacked bar per region over time
      if (ImGui::TreeNodeEx("Spike Raster",
              ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_SpanAvailWidth)) {
        ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
        ImVec2 canvas_size = ImVec2(ImGui::GetContentRegionAvail().x, 80.0f);
        ImGui::InvisibleButton("##raster", canvas_size);
        ImDrawList* draw = ImGui::GetWindowDrawList();

        float col_w = canvas_size.x / kRasterFrames;
        // Draw each frame as a column, brightness = total spike fraction
        for (int f = 0; f < kRasterFrames; ++f) {
          int idx = (raster_head + f) % kRasterFrames;
          float total = 0.0f;
          for (int r = 0; r < 13; ++r) total += raster_history[idx][r];
          total = std::clamp(total / 13.0f, 0.0f, 1.0f);
          float brightness = total * 3.0f; // amplify for visibility
          brightness = std::clamp(brightness, 0.0f, 1.0f);

          ImU32 color = ImGui::GetColorU32(
              ImVec4(brightness, brightness * 0.7f, brightness * 0.3f, 1.0f));
          float x0 = canvas_pos.x + f * col_w;
          draw->AddRectFilled(
              ImVec2(x0, canvas_pos.y),
              ImVec2(x0 + col_w, canvas_pos.y + canvas_size.y),
              color);
        }
        // Border
        draw->AddRect(canvas_pos,
                      ImVec2(canvas_pos.x + canvas_size.x,
                             canvas_pos.y + canvas_size.y),
                      ImGui::GetColorU32(ImGuiCol_Border));
        ImGui::TreePop();
      }

      // Motor output readout
      if (motor.HasMotorNeurons() &&
          ImGui::TreeNodeEx("Motor Output",
              ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_SpanAvailWidth)) {
        auto cmd = motor.Command();
        ImGui::Text("Forward:  %+.1f mm/s", cmd.forward_velocity);
        ImGui::Text("Angular:  %+.2f rad/s", cmd.angular_velocity);

        // Approach/avoid bar
        float drive = std::clamp(cmd.approach_drive, -5.0f, 5.0f);
        float norm = (drive + 5.0f) / 10.0f; // 0..1
        ImGui::Text("Valence:  %+.2f", cmd.approach_drive);
        ImGui::ProgressBar(norm, ImVec2(-1, 0),
            drive > 0.1f ? "Approach" : drive < -0.1f ? "Avoid" : "Neutral");

        if (cmd.freeze > 0.5f) {
          ImGui::TextColored(ImVec4(1, 0.5f, 0.2f, 1), "FREEZE");
        }
        ImGui::TreePop();
      }

      // TCP bridge to nmfly body
      ImGui::Separator();
      if (ImGui::Checkbox("TCP Server (nmfly)", &tcp_enabled)) {
        if (tcp_enabled) {
          if (!tcp_server.Start(9100))
            tcp_enabled = false;
        } else {
          tcp_server.Stop();
        }
      }
      if (tcp_enabled) {
        ImGui::SameLine();
        if (tcp_server.HasClient())
          ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.3f, 1), "Connected");
        else
          ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1), "Listening :9100");
      }

      // STDP toggle
      ImGui::Separator();
      ImGui::Checkbox("STDP Learning", &features.stdp);
      if (features.stdp) {
        ImGui::SetNextItemWidth(100);
        ImGui::SliderFloat("A+", &stdp_params.a_plus, 0.001f, 0.05f, "%.3f");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(100);
        ImGui::SliderFloat("tau", &stdp_params.tau_plus, 5.0f, 50.0f, "%.0f");
      }
      ImGui::Checkbox("Homeostasis", &features.homeostasis);
      ImGui::Checkbox("Gap Junctions", &features.gap_junctions);
      ImGui::Checkbox("Neuromodulation", &features.neuromodulation);
      ImGui::Checkbox("SFA (Adaptation)", &features.sfa);
      ImGui::Checkbox("Conduction Delays", &features.conduction_delays);
      ImGui::Checkbox("Short-Term Plasticity", &features.short_term_plasticity);
      ImGui::Checkbox("Structural Plasticity", &features.structural_plasticity);
      ImGui::Checkbox("Inhibitory Plasticity", &features.inhibitory_plasticity);
      ImGui::Checkbox("Neuromod Effects", &features.neuromodulator_effects);
      ImGui::Checkbox("NMDA Receptors", &features.nmda);

      ImGui::PopID();
    }

    // -- Regions (collapsing header) --
    // -- Help (collapsed by default, like Blender tooltips) --
    if (ImGui::CollapsingHeader("Controls", ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::Text("Fly");
      ImGui::TextDisabled("  W/A/S/D: walk / turn / stop");
      ImGui::Spacing();
      ImGui::Text("Brain Camera");
      ImGui::TextDisabled("  Orbit: left drag");
      ImGui::TextDisabled("  Pan: right drag");
      ImGui::TextDisabled("  Zoom: scroll");
      ImGui::TextDisabled("  R: reset");
    }

    ImGui::End();
    ImGui::PopStyleVar();
    } // show_panel

    // -- Brain panel: bottom-right, with transparent Regions overlay --
    if (body_window && brain_tex) {
      constexpr float kBrainW = 500.0f, kBrainH = 400.0f;
      int bwin_w, bwin_h;
      glfwGetWindowSize(body_window, &bwin_w, &bwin_h);

      ImGui::SetNextWindowPos(
          ImVec2(static_cast<float>(bwin_w) - kBrainW - 8.0f,
                 static_cast<float>(bwin_h) - kBrainH - 8.0f),
          ImGuiCond_Always);
      ImGui::SetNextWindowSize(ImVec2(kBrainW, kBrainH), ImGuiCond_Always);
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
      if (ImGui::Begin("Brain##viewport", nullptr,
                        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse |
                        ImGuiWindowFlags_NoMove)) {
        // Brain image fills entire panel
        ImVec2 avail = ImGui::GetContentRegionAvail();
        ImVec2 img_pos = ImGui::GetCursorScreenPos();
        int want_w = std::max(2, static_cast<int>(avail.x * 2));
        int want_h = std::max(2, static_cast<int>(avail.y * 2));
        if (want_w != brain_fbo_w || want_h != brain_fbo_h) {
          glfwMakeContextCurrent(window);
          CreateBrainFBO(want_w, want_h);
        }
        ImGui::Image((ImTextureID)(uintptr_t)brain_tex,
                     avail, ImVec2(0, 1), ImVec2(1, 0));

        // Mouse interaction on brain image
        if (ImGui::IsItemHovered()) {
          ImGuiIO& io = ImGui::GetIO();
          if (io.MouseWheel != 0.0f) {
            g_camera.radius *= (io.MouseWheel > 0) ? 0.9f : 1.1f;
            g_camera.radius = std::clamp(g_camera.radius, 50.0f, 2000.0f);
          }
          if (ImGui::IsKeyPressed(ImGuiKey_R)) g_camera.Reset();
          if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
            ImVec2 delta = io.MouseDelta;
            g_camera.azimuth += delta.x * 0.005f;
            g_camera.elevation += delta.y * 0.005f;
            g_camera.elevation = std::clamp(g_camera.elevation, -1.5f, 1.5f);
          }
          if (ImGui::IsMouseDragging(ImGuiMouseButton_Right) ||
              ImGui::IsMouseDragging(ImGuiMouseButton_Middle)) {
            ImVec2 delta = io.MouseDelta;
            float pan_speed = g_camera.radius * 0.002f;
            g_camera.target_x -= delta.x * pan_speed * std::cos(g_camera.azimuth);
            g_camera.target_z += delta.x * pan_speed * std::sin(g_camera.azimuth);
            g_camera.target_y += delta.y * pan_speed;
          }
        }

        // Transparent Regions overlay (top-left of brain image)
        static bool regions_open = false;
        ImGui::SetCursorScreenPos(ImVec2(img_pos.x + 4, img_pos.y + 4));
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4, 2));
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.1f, 0.15f, 0.7f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.2f, 0.2f, 0.3f, 0.8f));
        if (ImGui::SmallButton(regions_open ? "Regions [-]" : "Regions [+]"))
          regions_open = !regions_open;
        ImGui::PopStyleColor(2);
        ImGui::PopStyleVar();

        if (regions_open) {
          ImGui::SetCursorScreenPos(ImVec2(img_pos.x + 4, img_pos.y + 26));
          ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.08f, 0.08f, 0.12f, 0.85f));
          // Calculate height: button row (~26) + table rows * ~22 + padding
          // Collapsed: 8 top-level rows. Each expanded tree adds more.
          int vis_rows = 8; // base collapsed rows
          // Could count expanded trees, but a max cap works fine
          float reg_h = 26.0f + vis_rows * 22.0f + 12.0f; // ~208
          ImGui::BeginChild("##regions_overlay", ImVec2(300, reg_h), true);

          float tb = (ImGui::GetContentRegionAvail().x - 8) / 3.0f;
          if (ImGui::Button("All", ImVec2(tb, 0))) {
            for (int i = 0; i < 14; ++i) region_size[i] = 4.0f;
            region_dirty = true;
          }
          ImGui::SameLine();
          if (ImGui::Button("None", ImVec2(tb, 0))) {
            for (int i = 0; i < 14; ++i) region_size[i] = 0.0f;
            region_dirty = true;
          }
          ImGui::SameLine();
          if (ImGui::Button("Save", ImVec2(tb, 0)))
            SaveRegionConfig(config_path, region_size, sdf);

          const ImGuiTableFlags tflags =
              ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_NoPadOuterX;

          if (ImGui::BeginTable("##reg", 3, tflags)) {
            ImGui::TableSetupColumn("##vis",  ImGuiTableColumnFlags_WidthFixed, 20.0f);
            ImGui::TableSetupColumn("##name", ImGuiTableColumnFlags_WidthStretch, 1.0f);
            ImGui::TableSetupColumn("##size", ImGuiTableColumnFlags_WidthStretch, 0.7f);

            auto eye_swatch = [&](int idx) {
              ImGui::TableNextColumn();
              float cr, cg, cb;
              RegionColor(idx, cr, cg, cb);
              bool visible = region_size[idx] > 0.0f;
              float alpha = visible ? 1.0f : 0.25f;
              if (ImGui::ColorButton(("##e" + std::to_string(idx)).c_str(),
                                     ImVec4(cr, cg, cb, alpha),
                                     ImGuiColorEditFlags_NoTooltip, ImVec2(14, 14))) {
                region_size[idx] = visible ? 0.0f : 4.0f;
                region_dirty = true;
              }
              if (ImGui::IsItemHovered())
                ImGui::SetTooltip(visible ? "Hide" : "Show");
            };

            auto leaf = [&](const char* label, int idx) {
              ImGui::TableNextRow();
              eye_swatch(idx);
              ImGui::TableNextColumn();
              ImGui::AlignTextToFramePadding();
              ImGui::Text("%s", label);
              ImGui::TableNextColumn();
              ImGui::SetNextItemWidth(-1);
              if (ImGui::SliderFloat(("##s" + std::to_string(idx)).c_str(),
                                     &region_size[idx], 0.0f, 12.0f, "%.0f"))
                region_dirty = true;
            };

            auto group_slider = [&](const char* id, const int* indices, int n) {
              ImGui::TableNextColumn();
              float sum = 0;
              for (int i = 0; i < n; ++i) sum += region_size[indices[i]];
              float avg = sum / static_cast<float>(n);
              ImGui::SetNextItemWidth(-1);
              if (ImGui::SliderFloat(id, &avg, 0.0f, 12.0f, "%.0f")) {
                for (int i = 0; i < n; ++i) region_size[indices[i]] = avg;
                region_dirty = true;
              }
            };

            auto group_eye = [&](const char* id, int color_idx,
                                 const int* indices, int n) {
              ImGui::TableNextColumn();
              float cr, cg, cb;
              RegionColor(color_idx, cr, cg, cb);
              bool any_vis = false;
              for (int i = 0; i < n; ++i)
                if (region_size[indices[i]] > 0.0f) any_vis = true;
              float alpha = any_vis ? 1.0f : 0.25f;
              if (ImGui::ColorButton(id, ImVec4(cr, cg, cb, alpha),
                                     ImGuiColorEditFlags_NoTooltip, ImVec2(14, 14))) {
                float val = any_vis ? 0.0f : 4.0f;
                for (int i = 0; i < n; ++i) region_size[indices[i]] = val;
                region_dirty = true;
              }
            };

            auto pair = [&](const char* name, int l, int r) {
              int idx[2] = {l, r};
              ImGui::TableNextRow();
              group_eye(("##ge" + std::to_string(l)).c_str(), l, idx, 2);
              ImGui::TableNextColumn();
              bool open = ImGui::TreeNodeEx(name, ImGuiTreeNodeFlags_SpanAvailWidth);
              group_slider(("##a" + std::to_string(l)).c_str(), idx, 2);
              if (open) {
                leaf("L", l);
                leaf("R", r);
                ImGui::TreePop();
              }
            };

            leaf("Protocerebrum", 0);
            pair("Optic Lobe", 1, 2);
            {
              static const int mb[] = {3, 4, 5, 6};
              ImGui::TableNextRow();
              group_eye("##ge_mb", 3, mb, 4);
              ImGui::TableNextColumn();
              bool open = ImGui::TreeNodeEx("Mushroom Body",
                  ImGuiTreeNodeFlags_SpanAvailWidth);
              group_slider("##cat_mb", mb, 4);
              if (open) {
                pair("Calyx", 3, 4);
                pair("Lobe", 5, 6);
                ImGui::TreePop();
              }
            }
            pair("Antennal Lobe", 7, 8);
            leaf("Central Complex", 9);
            pair("Lateral Horn", 10, 11);
            leaf("Subesophageal Zone", 12);

            ImGui::EndTable();
          }

          ImGui::EndChild();
          ImGui::PopStyleColor();
        }
      }
      ImGui::End();
      ImGui::PopStyleVar();
    }

    // Update per-vertex sizes based on region sliders
    if (region_dirty) {
      region_dirty = false;
      for (size_t i = 0; i < points.size(); ++i) {
        points[i].size = region_size[points[i].region];
      }
      glBindBuffer(GL_ARRAY_BUFFER, vbo);
      glBufferSubData(GL_ARRAY_BUFFER, 0,
                      static_cast<GLsizeiptr>(points.size() * sizeof(PointVertex)),
                      points.data());
    }

    // (Fly sandbox arena panel removed: was dead code behind if(false))

    ImGui::Render();

    // --- Brain: render point cloud to FBO ---
    glfwMakeContextCurrent(window);
    {
      glBindFramebuffer(GL_FRAMEBUFFER, brain_fbo);
      glViewport(0, 0, brain_fbo_w, brain_fbo_h);
      glClearColor(0.06f, 0.06f, 0.10f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      glEnable(GL_DEPTH_TEST);
      glEnable(GL_PROGRAM_POINT_SIZE);
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

      float aspect = brain_fbo_w > 0 ? static_cast<float>(brain_fbo_w) / brain_fbo_h : 1.0f;
      Mat4 proj = Mat4::Perspective(0.785f, aspect, 1.0f, 5000.0f);
      Mat4 view = g_camera.ViewMatrix();

      glUseProgram(program);
      glUniformMatrix4fv(loc_proj, 1, GL_FALSE, proj.m);
      glUniformMatrix4fv(loc_view, 1, GL_FALSE, view.m);

      glBindVertexArray(vao);
      glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(points.size()));
      glBindVertexArray(0);
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
      glFlush(); // ensure texture is ready before body context reads it
    }

#ifdef FWMC_BODY_SIM
    // --- Body window: render MuJoCo + ImGui overlay ---
    if (body_window && g_body.ready && !glfwWindowShouldClose(body_window)) {
      glfwMakeContextCurrent(body_window);
      int bw, bh;
      glfwGetFramebufferSize(body_window, &bw, &bh);
      glViewport(0, 0, bw, bh);
      glClearColor(0.2f, 0.2f, 0.3f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      mjrRect mj_vp = {0, 0, bw, bh};
      g_body.Render(mj_vp);
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
      glViewport(0, 0, bw, bh);
      ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
      glfwSwapBuffers(body_window);

      // MuJoCo camera mouse controls on body window.
      int bsw, bsh;
      glfwGetWindowSize(body_window, &bsw, &bsh);
      double mx, my;
      glfwGetCursorPos(body_window, &mx, &my);
      bool left = glfwGetMouseButton(body_window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
      bool right = glfwGetMouseButton(body_window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
      bool mid = glfwGetMouseButton(body_window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;

      if (left && !g_body_mouse_left) { g_body_mouse_left = true; g_body_last_mx = mx; g_body_last_my = my; }
      if (!left) g_body_mouse_left = false;
      if (right && !g_body_mouse_right) { g_body_mouse_right = true; g_body_last_mx = mx; g_body_last_my = my; }
      if (!right) g_body_mouse_right = false;
      if (mid && !g_body_mouse_mid) { g_body_mouse_mid = true; g_body_last_mx = mx; g_body_last_my = my; }
      if (!mid) g_body_mouse_mid = false;

      if (!ImGui::GetIO().WantCaptureMouse) {
        double dx = mx - g_body_last_mx, dy = my - g_body_last_my;
        if (g_body_mouse_left) g_body.RotateCamera(dx, dy, bsw, bsh);
        else if (g_body_mouse_right) g_body.TranslateCamera(dx, dy, bsw, bsh);
        else if (g_body_mouse_mid) g_body.ZoomCamera(dy, bsh);
      }
      g_body_last_mx = mx; g_body_last_my = my;
      glfwMakeContextCurrent(window);
    }
#endif

    // --- Brain-only mode: render FBO to screen ---
    if (!body_window) {
      int fb_w, fb_h;
      glfwGetFramebufferSize(window, &fb_w, &fb_h);
      glViewport(0, 0, fb_w, fb_h);
      glClearColor(0.06f, 0.06f, 0.10f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      // Blit FBO to screen
      glBindFramebuffer(GL_READ_FRAMEBUFFER, brain_fbo);
      glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
      glBlitFramebuffer(0, 0, brain_fbo_w, brain_fbo_h,
                        0, 0, fb_w, fb_h,
                        GL_COLOR_BUFFER_BIT, GL_LINEAR);
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
      ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
      glfwSwapBuffers(window);
    }
  }

  // Cleanup
  glfwMakeContextCurrent(window);
  glDeleteFramebuffers(1, &brain_fbo);
  glDeleteTextures(1, &brain_tex);
  glDeleteRenderbuffers(1, &brain_depth);
  glDeleteVertexArrays(1, &vao);
  glDeleteBuffers(1, &vbo);
  glDeleteProgram(program);
  tcp_server.Stop();
  if (body_window) {
    glfwMakeContextCurrent(body_window);
    g_body.Shutdown();
  }
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  if (body_window) glfwDestroyWindow(body_window);
  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}
