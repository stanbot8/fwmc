#ifndef FWMC_BRAIN_SDF_H_
#define FWMC_BRAIN_SDF_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "tissue/voxel_grid.h"

namespace fwmc {

// Signed distance field primitives for constructing brain geometry.
// Negative = inside, positive = outside, zero = surface.
// The brain shape emerges from compositing ellipsoid primitives via
// smooth CSG unions, then refining with PDE-based smoothing (diffusion
// of the distance field itself, which rounds sharp junctions).

struct SDFPrimitive {
  std::string name;      // anatomical label
  float cx, cy, cz;      // center (um)
  float rx, ry, rz;      // radii (um)

  // Signed distance to an axis-aligned ellipsoid.
  // Uses the scaling trick: transform to unit sphere, compute distance,
  // scale back. Not exact SDF but good enough for smooth union.
  float Evaluate(float x, float y, float z) const {
    float dx = (x - cx) / rx;
    float dy = (y - cy) / ry;
    float dz = (z - cz) / rz;
    float r = std::sqrt(dx*dx + dy*dy + dz*dz);
    // Approximate SDF: scale by minimum radius for reasonable gradients
    float min_r = std::min({rx, ry, rz});
    return (r - 1.0f) * min_r;
  }
};

// Smooth minimum (polynomial). Blends two SDF values with smooth
// transition of width k. This creates organic-looking junctions
// between brain regions instead of sharp creases.
inline float SmoothMin(float a, float b, float k) {
  float h = std::max(k - std::abs(a - b), 0.0f) / k;
  return std::min(a, b) - h * h * h * k * (1.0f / 6.0f);
}

// Drosophila brain shape as a composite SDF.
// Built from ellipsoid primitives representing the major neuropil regions.
// Dimensions based on adult Drosophila melanogaster brain atlas
// (Ito et al. 2014, ~500um wide, ~300um tall, ~200um deep).
struct BrainSDF {
  std::vector<SDFPrimitive> primitives;
  float smooth_k = 15.0f;  // smooth union blend radius (um)

  // Initialize with Drosophila brain anatomy.
  // All coordinates in micrometers, centered at (250, 150, 100).
  void InitDrosophila() {
    primitives.clear();

    // Central brain (protocerebrum): the main mass
    primitives.push_back({"central_brain", 250, 150, 100, 120, 90, 70});

    // Optic lobes: large lateral structures for visual processing
    primitives.push_back({"optic_lobe_L", 80, 150, 100, 70, 80, 60});
    primitives.push_back({"optic_lobe_R", 420, 150, 100, 70, 80, 60});

    // Mushroom bodies: learning/memory centers, dorsal-posterior
    // Calyx (input region) is larger, pedunculus is elongated
    primitives.push_back({"mb_calyx_L", 180, 200, 120, 35, 30, 25});
    primitives.push_back({"mb_calyx_R", 320, 200, 120, 35, 30, 25});
    primitives.push_back({"mb_lobe_L", 180, 130, 80, 20, 40, 20});
    primitives.push_back({"mb_lobe_R", 320, 130, 80, 20, 40, 20});

    // Antennal lobes: olfactory processing, anterior-ventral
    primitives.push_back({"antennal_lobe_L", 200, 90, 60, 30, 25, 25});
    primitives.push_back({"antennal_lobe_R", 300, 90, 60, 30, 25, 25});

    // Central complex: navigation/motor coordination, midline
    primitives.push_back({"central_complex", 250, 170, 100, 30, 15, 15});

    // Lateral horn: innate olfactory behavior, lateral to MB
    primitives.push_back({"lateral_horn_L", 150, 180, 110, 25, 20, 20});
    primitives.push_back({"lateral_horn_R", 350, 180, 110, 25, 20, 20});

    // Subesophageal zone: gustatory/motor, ventral
    primitives.push_back({"sez", 250, 70, 90, 60, 40, 40});
  }

  // Evaluate the composite SDF at a point.
  // Smooth union of all primitives.
  float Evaluate(float x, float y, float z) const {
    if (primitives.empty()) return 1.0f;
    float d = primitives[0].Evaluate(x, y, z);
    for (size_t i = 1; i < primitives.size(); ++i) {
      d = SmoothMin(d, primitives[i].Evaluate(x, y, z), smooth_k);
    }
    return d;
  }

  // Which region is closest to a point? Returns index into primitives,
  // or -1 if outside all regions.
  int NearestRegion(float x, float y, float z) const {
    // Among all primitives the point is inside, pick the smallest
    // (most specific) one. This prevents large regions like central_brain
    // from swallowing smaller sub-regions like mushroom body lobes.
    int best = -1;
    float best_vol = 1e30f;
    float closest_d = 1e30f;
    for (size_t i = 0; i < primitives.size(); ++i) {
      auto& p = primitives[i];
      float d = p.Evaluate(x, y, z);
      if (d <= 0.0f) {
        // Point is inside this primitive; prefer smaller volume
        float vol = p.rx * p.ry * p.rz;
        if (vol < best_vol) {
          best_vol = vol;
          best = static_cast<int>(i);
        }
      } else if (best < 0 && d < closest_d) {
        // Fallback: if not inside any, track the closest
        closest_d = d;
        best = static_cast<int>(i);
      }
    }
    return (best_vol < 1e30f) ? best : -1;
  }

  // Bake the SDF into a VoxelGrid channel.
  // Also creates a "region_id" channel mapping each voxel to its
  // nearest anatomical region (or -1 if outside).
  void BakeToGrid(VoxelGrid& grid, size_t sdf_channel,
                  size_t region_channel) const {
    for (uint32_t z = 0; z < grid.nz; ++z) {
      for (uint32_t y = 0; y < grid.ny; ++y) {
        for (uint32_t x = 0; x < grid.nx; ++x) {
          float wx, wy, wz;
          grid.GridToWorld(x, y, z, wx, wy, wz);
          size_t idx = grid.Idx(x, y, z);
          grid.channels[sdf_channel].data[idx] = Evaluate(wx, wy, wz);
          grid.channels[region_channel].data[idx] =
              static_cast<float>(NearestRegion(wx, wy, wz));
        }
      }
    }
  }

  // Smooth the baked SDF using diffusion (Laplacian smoothing).
  // This rounds sharp junctions between primitives, making the
  // brain surface organic rather than a union of lumps.
  // Operates only on voxels near the surface (|sdf| < band).
  static void DiffuseSmooth(VoxelGrid& grid, size_t sdf_channel,
                            int iterations, float band_um = 30.0f) {
    auto& data = grid.channels[sdf_channel].data;
    std::vector<float> temp(data.size());

    for (int iter = 0; iter < iterations; ++iter) {
      for (uint32_t z = 0; z < grid.nz; ++z) {
        for (uint32_t y = 0; y < grid.ny; ++y) {
          for (uint32_t x = 0; x < grid.nx; ++x) {
            size_t i = grid.Idx(x, y, z);
            float c = data[i];

            // Only smooth near the surface
            if (std::abs(c) > band_um) {
              temp[i] = c;
              continue;
            }

            float xm = (x > 0)          ? data[grid.Idx(x-1,y,z)] : c;
            float xp = (x < grid.nx-1)  ? data[grid.Idx(x+1,y,z)] : c;
            float ym = (y > 0)          ? data[grid.Idx(x,y-1,z)] : c;
            float yp = (y < grid.ny-1)  ? data[grid.Idx(x,y+1,z)] : c;
            float zm = (z > 0)          ? data[grid.Idx(x,y,z-1)] : c;
            float zp = (z < grid.nz-1)  ? data[grid.Idx(x,y,z+1)] : c;

            // Laplacian smoothing (1/6 weight to neighbors)
            temp[i] = (xm + xp + ym + yp + zm + zp) / 6.0f;
          }
        }
      }
      data.swap(temp);
    }
  }

  // Compute surface normal at a point via central differences on the SDF.
  void Normal(float x, float y, float z, float eps,
              float& nx_out, float& ny_out, float& nz_out) const {
    float gx = Evaluate(x + eps, y, z) - Evaluate(x - eps, y, z);
    float gy = Evaluate(x, y + eps, z) - Evaluate(x, y - eps, z);
    float gz = Evaluate(x, y, z + eps) - Evaluate(x, y, z - eps);
    float len = std::sqrt(gx*gx + gy*gy + gz*gz);
    if (len > 1e-10f) { gx /= len; gy /= len; gz /= len; }
    nx_out = gx; ny_out = gy; nz_out = gz;
  }
};

}  // namespace fwmc

#endif  // FWMC_BRAIN_SDF_H_
