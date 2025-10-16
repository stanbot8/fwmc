#ifndef FWMC_MUJOCO_FLY_H_
#define FWMC_MUJOCO_FLY_H_

// Embedded NeuroMechFly body driven by brain motor output.
//
// Loads the NMF MuJoCo model, runs a tripod CPG driven by MotorCommand,
// renders offscreen via a hidden compatibility-profile GLFW window,
// and provides the framebuffer as an OpenGL texture for ImGui display.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <array>
#include <string>
#include <vector>

#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>
#include <glad/gl.h>

#include "core/motor_output.h"

namespace fwmc {

// Tripod gait CPG (port of nmfly FlygymGait)
struct TripodCPG {
  static constexpr int kLegs = 6;
  static constexpr int kJointsPerLeg = 7;
  static constexpr int kJoints = 42;

  float phases[6] = {0, 3.14159265f, 0, 3.14159265f, 0, 3.14159265f};
  float coupling[6][6] = {};
  float rest_pose[42] = {};

  float fwd_vel = 0.0f;
  float ang_vel = 0.0f;
  bool frozen = false;
  float tau = 0.05f;

  void Init(const float* rest, int n) {
    int count = n < kJoints ? n : kJoints;
    std::memcpy(rest_pose, rest, count * sizeof(float));

    const float pi = 3.14159265f;
    auto in_a = [](int i) { return i == 0 || i == 2 || i == 4; };
    for (int i = 0; i < 6; ++i)
      for (int j = 0; j < 6; ++j)
        coupling[i][j] = (i == j) ? 0.0f : (in_a(i) == in_a(j) ? 0.0f : pi);
  }

  void Update(const MotorCommand& cmd, float dt) {
    float alpha = 1.0f - std::exp(-dt / tau);
    fwd_vel += alpha * (cmd.forward_velocity - fwd_vel);
    ang_vel += alpha * (cmd.angular_velocity - ang_vel);
    frozen = cmd.freeze > 0.5f;
    if (frozen) return;

    float speed = std::abs(fwd_vel);
    float freq = std::clamp(speed * 0.4f, 2.0f, 15.0f);
    float dir = fwd_vel >= 0 ? 1.0f : -1.0f;
    const float two_pi = 6.28318530f;

    float dp[6];
    for (int i = 0; i < 6; ++i) dp[i] = dir * two_pi * freq;
    for (int i = 0; i < 6; ++i)
      for (int j = 0; j < 6; ++j)
        if (i != j)
          dp[i] += 10.0f * std::sin(phases[j] - phases[i] - coupling[i][j]);

    for (int i = 0; i < 6; ++i) {
      phases[i] += dp[i] * dt;
      phases[i] = std::fmod(phases[i], two_pi);
      if (phases[i] < 0) phases[i] += two_pi;
    }
  }

  void GetTargets(float* targets) const {
    std::memcpy(targets, rest_pose, kJoints * sizeof(float));
    if (frozen) return;

    float ss = std::min(1.0f, std::abs(fwd_vel) / 30.0f);
    for (int leg = 0; leg < kLegs; ++leg) {
      float sw = std::sin(phases[leg]);
      float li = std::max(0.0f, sw);
      bool left = leg < 3;
      float tb = 1.0f;
      if (ang_vel > 0.1f) tb = left ? 0.5f : 1.5f;
      else if (ang_vel < -0.1f) tb = left ? 1.5f : 0.5f;
      int b = leg * kJointsPerLeg;
      float a = ss * tb;
      targets[b + 0] += a * sw * 0.3f;
      targets[b + 3] += a * li * 0.4f;
      targets[b + 5] += a * li * 0.2f;
    }
  }
};

// MuJoCo fly: physics + offscreen rendering via hidden compat window
struct MujocoFly {
  mjModel* model = nullptr;
  mjData* data = nullptr;
  mjvScene scene = {};
  mjvCamera camera = {};
  mjvOption opt = {};
  mjrContext mj_ctx = {};

  TripodCPG cpg;
  bool loaded = false;
  int render_width = 640;
  int render_height = 480;
  GLuint texture_id = 0;
  std::vector<uint8_t> rgb_buffer;

  // Hidden GLFW window for MuJoCo's GL context
  GLFWwindow* offscreen_window = nullptr;
  // Main viewer window (to restore context after MuJoCo render)
  GLFWwindow* main_window = nullptr;

  static constexpr int kJointActuators = 42;
  static constexpr int kAdhesionStart = 42;

  bool Load(const char* model_path, GLFWwindow* main_win) {
    main_window = main_win;
    char error[1000] = {};

    std::string path(model_path);
    if (path.size() > 4 && path.substr(path.size() - 4) == ".mjb") {
      model = mj_loadModel(model_path, nullptr);
    } else {
      model = mj_loadXML(model_path, nullptr, error, sizeof(error));
    }
    if (!model) {
      fprintf(stderr, "MuJoCo load failed: %s\n", error);
      return false;
    }

    data = mj_makeData(model);
    if (!data) {
      mj_deleteModel(model);
      model = nullptr;
      return false;
    }

    // Set offscreen buffer size to match our render resolution
    model->vis.global.offwidth = render_width;
    model->vis.global.offheight = render_height;

    // Create hidden GLFW window with compatibility profile for MuJoCo
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_ANY_PROFILE);
    offscreen_window = glfwCreateWindow(render_width, render_height,
                                        "mujoco_offscreen", nullptr, nullptr);
    if (!offscreen_window) {
      fprintf(stderr, "Failed to create MuJoCo offscreen window\n");
      mj_deleteData(data); data = nullptr;
      mj_deleteModel(model); model = nullptr;
      return false;
    }

    // Switch to MuJoCo's GL context
    glfwMakeContextCurrent(offscreen_window);

    // Initialize MuJoCo visualization in the compat context
    mjv_defaultScene(&scene);
    mjv_defaultCamera(&camera);
    mjv_defaultOption(&opt);
    mjr_defaultContext(&mj_ctx);
    mjv_makeScene(model, &scene, 2000);
    mjr_makeContext(model, &mj_ctx, mjFONTSCALE_100);

    // Use MuJoCo's offscreen framebuffer (not the window's)
    mjr_setBuffer(mjFB_OFFSCREEN, &mj_ctx);

    camera.type = mjCAMERA_FREE;
    camera.lookat[0] = 0;
    camera.lookat[1] = 0;
    camera.lookat[2] = 0.3;
    camera.distance = 4.0;
    camera.azimuth = 180;
    camera.elevation = -20;

    // Switch back to main viewer context
    glfwMakeContextCurrent(main_window);

    // Reset GLFW hints for any future window creation
    glfwDefaultWindowHints();

    // Reset sim and capture rest pose
    mj_resetData(model, data);
    mj_forward(model, data);

    float rest[42];
    for (int i = 0; i < kJointActuators && i < model->nu; ++i) {
      int jid = model->actuator_trnid[2 * i];
      int qa = model->jnt_qposadr[jid];
      rest[i] = static_cast<float>(data->qpos[qa]);
    }
    cpg.Init(rest, kJointActuators);

    // Allocate RGB buffer
    rgb_buffer.resize(render_width * render_height * 3);

    // Create texture in main GL context
    glGenTextures(1, &texture_id);
    glBindTexture(GL_TEXTURE_2D, texture_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, render_width, render_height,
                 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);

    loaded = true;
    printf("NMF model loaded: nq=%d nv=%d nu=%d\n",
           model->nq, model->nv, model->nu);
    return true;
  }

  void Step(const MotorCommand& cmd, float dt_sec, int substeps = 5) {
    if (!loaded) return;

    cpg.Update(cmd, dt_sec * substeps);

    float targets[42];
    cpg.GetTargets(targets);

    for (int i = 0; i < kJointActuators && i < model->nu; ++i)
      data->ctrl[i] = targets[i];
    for (int i = kAdhesionStart; i < model->nu; ++i)
      data->ctrl[i] = 1.0;

    for (int i = 0; i < substeps; ++i)
      mj_step(model, data);
  }

  void Render() {
    if (!loaded || !offscreen_window) return;

    // Switch to MuJoCo's compat GL context
    glfwMakeContextCurrent(offscreen_window);

    mjv_updateScene(model, data, &opt, nullptr, &camera,
                    mjCAT_ALL, &scene);

    mjrRect vp = {0, 0, render_width, render_height};
    mjr_render(vp, &scene, &mj_ctx);
    mjr_readPixels(rgb_buffer.data(), nullptr, vp, &mj_ctx);

    // Switch back to main viewer context
    glfwMakeContextCurrent(main_window);

    // Flip vertically (MuJoCo renders bottom-up)
    int row_bytes = render_width * 3;
    std::vector<uint8_t> row(row_bytes);
    for (int y = 0; y < render_height / 2; ++y) {
      int top = y * row_bytes;
      int bot = (render_height - 1 - y) * row_bytes;
      std::memcpy(row.data(), rgb_buffer.data() + top, row_bytes);
      std::memcpy(rgb_buffer.data() + top, rgb_buffer.data() + bot, row_bytes);
      std::memcpy(rgb_buffer.data() + bot, row.data(), row_bytes);
    }

    // Upload to texture
    glBindTexture(GL_TEXTURE_2D, texture_id);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, render_width, render_height,
                    GL_RGB, GL_UNSIGNED_BYTE, rgb_buffer.data());
  }

  void Cleanup() {
    if (loaded) {
      if (offscreen_window) {
        glfwMakeContextCurrent(offscreen_window);
        mjr_freeContext(&mj_ctx);
        mjv_freeScene(&scene);
        glfwMakeContextCurrent(main_window);
        glfwDestroyWindow(offscreen_window);
        offscreen_window = nullptr;
      }
      if (texture_id) {
        glDeleteTextures(1, &texture_id);
        texture_id = 0;
      }
      mj_deleteData(data); data = nullptr;
      mj_deleteModel(model); model = nullptr;
      loaded = false;
    }
  }

  ~MujocoFly() { Cleanup(); }
};

}  // namespace fwmc

#endif  // FWMC_MUJOCO_FLY_H_
