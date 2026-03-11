// Tissue module tests: voxel grid, brain SDF, neural field, LOD manager
#include "test_harness.h"

#include "tissue/voxel_grid.h"
#include "tissue/brain_sdf.h"
#include "tissue/neural_field.h"
#include "tissue/lod_manager.h"

#include <cmath>

// ===== VoxelGrid tests =====

TEST(voxel_grid_init) {
  VoxelGrid grid;
  grid.Init(10, 10, 10, 5.0f);
  assert(grid.nx == 10);
  assert(grid.ny == 10);
  assert(grid.nz == 10);
  assert(grid.NumVoxels() == 1000);
  assert(grid.dx == 5.0f);
}

TEST(voxel_grid_add_channel) {
  VoxelGrid grid;
  grid.Init(5, 5, 5, 1.0f);
  size_t ch = grid.AddChannel("test", 0.1f, 0.01f);
  assert(ch == 0);
  assert(grid.channels[ch].data.size() == 125);
  assert(grid.channels[ch].diffusion_coeff == 0.1f);
  assert(grid.channels[ch].decay_rate == 0.01f);
  assert(grid.FindChannel("test") == 0);
  assert(grid.FindChannel("nonexistent") == SIZE_MAX);
}

TEST(voxel_grid_world_to_grid) {
  VoxelGrid grid;
  grid.Init(10, 10, 10, 5.0f);
  grid.origin_x = 0; grid.origin_y = 0; grid.origin_z = 0;

  uint32_t gx = 0, gy = 0, gz = 0;
  bool ok = grid.WorldToGrid(12.0f, 22.0f, 7.0f, gx, gy, gz);
  assert(ok);
  assert(gx == 2);
  assert(gy == 4);
  assert(gz == 1);
  (void)ok; (void)gx; (void)gy; (void)gz;

  // Outside the grid (reuse gx/gy/gz)
  bool out1 = grid.WorldToGrid(-1.0f, 0.0f, 0.0f, gx, gy, gz);
  assert(!out1); (void)out1;
  bool out2 = grid.WorldToGrid(60.0f, 0.0f, 0.0f, gx, gy, gz);
  assert(!out2); (void)out2;
}

TEST(voxel_grid_inject_and_sample) {
  VoxelGrid grid;
  grid.Init(10, 10, 10, 5.0f);
  size_t ch = grid.AddChannel("conc");

  grid.Inject(ch, 12.5f, 12.5f, 12.5f, 100.0f);

  // Sample at injection point should be high
  float val = grid.Sample(ch, 12.5f, 12.5f, 12.5f);
  assert(val > 0.0f);

  // Sample far away should be zero
  float far = grid.Sample(ch, 42.5f, 42.5f, 42.5f);
  assert(far == 0.0f);
}

TEST(voxel_grid_diffusion) {
  VoxelGrid grid;
  grid.Init(20, 20, 20, 1.0f);
  size_t ch = grid.AddChannel("heat", 1.0f, 0.0f);

  // Point source at center
  grid.channels[ch].data[grid.Idx(10, 10, 10)] = 100.0f;

  float center_before = grid.channels[ch].data[grid.Idx(10, 10, 10)];
  float neighbor_before = grid.channels[ch].data[grid.Idx(11, 10, 10)];

  // Diffuse
  grid.Diffuse(0.1f);

  float center_after = grid.channels[ch].data[grid.Idx(10, 10, 10)];
  float neighbor_after = grid.channels[ch].data[grid.Idx(11, 10, 10)];

  // Center should decrease, neighbor should increase
  assert(center_after < center_before);
  assert(neighbor_after > neighbor_before);
}

TEST(voxel_grid_decay) {
  VoxelGrid grid;
  grid.Init(5, 5, 5, 1.0f);
  size_t ch = grid.AddChannel("substance", 0.0f, 0.1f);  // decay only

  // Fill with uniform value
  for (auto& v : grid.channels[ch].data) v = 10.0f;

  grid.Diffuse(1.0f);  // triggers decay

  // Should be reduced by exp(-0.1)
  float expected = 10.0f * std::exp(-0.1f);
  for (auto& v : grid.channels[ch].data) {
    assert(std::abs(v - expected) < 0.01f);
  }
}

TEST(voxel_grid_inject_sphere) {
  VoxelGrid grid;
  grid.Init(20, 20, 20, 1.0f);
  size_t ch = grid.AddChannel("sphere");

  grid.InjectSphere(ch, 10.0f, 10.0f, 10.0f, 3.0f, 100.0f);

  // Center voxel should have some value
  float center = grid.channels[ch].data[grid.Idx(10, 10, 10)];
  assert(center > 0.0f);

  // Far corner should be zero
  float corner = grid.channels[ch].data[grid.Idx(0, 0, 0)];
  assert(corner == 0.0f);

  // Total injected should be ~100 (distributed across voxels in sphere)
  float total = 0.0f;
  for (auto v : grid.channels[ch].data) total += v;
  assert(std::abs(total - 100.0f) < 0.1f);
}

// ===== BrainSDF tests =====

TEST(brain_sdf_single_sphere) {
  BrainSDF sdf;
  sdf.primitives.push_back({"sphere", 50, 50, 50, 30, 30, 30});

  // Center should be inside (negative)
  float d_center = sdf.Evaluate(50, 50, 50);
  assert(d_center < 0.0f);

  // Far outside should be positive
  float d_outside = sdf.Evaluate(200, 200, 200);
  assert(d_outside > 0.0f);

  // On the surface should be near zero
  float d_surface = sdf.Evaluate(80, 50, 50);
  assert(std::abs(d_surface) < 5.0f);
}

TEST(brain_sdf_smooth_union) {
  BrainSDF sdf;
  sdf.smooth_k = 10.0f;
  sdf.primitives.push_back({"A", 40, 50, 50, 20, 20, 20});
  sdf.primitives.push_back({"B", 70, 50, 50, 20, 20, 20});

  // Both centers should be inside
  assert(sdf.Evaluate(40, 50, 50) < 0.0f);
  assert(sdf.Evaluate(70, 50, 50) < 0.0f);

  // The junction between them (at x=55) should also be inside
  // due to smooth union blending
  float d_junction = sdf.Evaluate(55, 50, 50);
  assert(d_junction < 0.0f);
}

TEST(brain_sdf_drosophila_init) {
  BrainSDF sdf;
  sdf.InitDrosophila();

  // Should have primitives for all major regions
  assert(sdf.primitives.size() >= 10);

  // Center of brain should be inside
  assert(sdf.Evaluate(250, 150, 100) < 0.0f);

  // Far outside should be positive
  assert(sdf.Evaluate(0, 0, 0) > 0.0f);
  assert(sdf.Evaluate(600, 400, 300) > 0.0f);

  // Optic lobes should be inside
  assert(sdf.Evaluate(80, 150, 100) < 0.0f);   // left
  assert(sdf.Evaluate(420, 150, 100) < 0.0f);  // right
}

TEST(brain_sdf_nearest_region) {
  BrainSDF sdf;
  sdf.InitDrosophila();

  // Center of brain = central_brain (index 0)
  int reg = sdf.NearestRegion(250, 150, 100);
  assert(reg == 0);

  // Left optic lobe center
  int lobe = sdf.NearestRegion(80, 150, 100);
  assert(lobe == 1);  // optic_lobe_L is index 1

  // Outside should return -1
  int outside = sdf.NearestRegion(0, 0, 0);
  assert(outside == -1);
}

TEST(brain_sdf_bake_and_smooth) {
  BrainSDF sdf;
  sdf.primitives.push_back({"sphere", 25, 25, 25, 15, 15, 15});

  VoxelGrid grid;
  grid.Init(10, 10, 10, 5.0f);
  size_t ch_sdf = grid.AddChannel("sdf");
  size_t ch_reg = grid.AddChannel("region");

  sdf.BakeToGrid(grid, ch_sdf, ch_reg);

  // Center voxel should be inside (negative SDF)
  float center_sdf = grid.channels[ch_sdf].data[grid.Idx(5, 5, 5)];
  assert(center_sdf < 0.0f);

  // Corner should be outside (positive SDF)
  float corner_sdf = grid.channels[ch_sdf].data[grid.Idx(0, 0, 0)];
  assert(corner_sdf > 0.0f);

  // Smooth the SDF
  float pre_smooth = grid.channels[ch_sdf].data[grid.Idx(3, 5, 5)];
  BrainSDF::DiffuseSmooth(grid, ch_sdf, 5);
  float post_smooth = grid.channels[ch_sdf].data[grid.Idx(3, 5, 5)];

  // Smoothing should change surface-adjacent values
  // (the exact change depends on geometry, just verify it's different)
  assert(pre_smooth != post_smooth || pre_smooth == 0.0f);
}

TEST(brain_sdf_normal) {
  BrainSDF sdf;
  sdf.primitives.push_back({"sphere", 50, 50, 50, 30, 30, 30});

  float nx, ny, nz;
  // Normal on the +x surface should point roughly in +x
  sdf.Normal(80, 50, 50, 1.0f, nx, ny, nz);
  assert(nx > 0.5f);
  assert(std::abs(ny) < 0.3f);
  assert(std::abs(nz) < 0.3f);
}

// ===== NeuralField tests =====

TEST(neural_field_init) {
  VoxelGrid grid;
  grid.Init(10, 10, 10, 5.0f);
  size_t ch_sdf = grid.AddChannel("sdf");

  // Create a simple spherical brain
  BrainSDF sdf;
  sdf.primitives.push_back({"brain", 25, 25, 25, 20, 20, 20});
  size_t ch_reg = grid.AddChannel("region");
  sdf.BakeToGrid(grid, ch_sdf, ch_reg);

  NeuralField field;
  field.ch_sdf = ch_sdf;
  field.Init(grid);

  // E and I channels should exist
  assert(field.ch_e != SIZE_MAX);
  assert(field.ch_i != SIZE_MAX);

  // Inside voxels should have initial activity
  float center_e = grid.channels[field.ch_e].data[grid.Idx(5, 5, 5)];
  assert(center_e > 0.0f);

  // Outside voxels should be zero
  float corner_e = grid.channels[field.ch_e].data[grid.Idx(0, 0, 0)];
  assert(corner_e == 0.0f);
}

TEST(neural_field_step_evolves) {
  VoxelGrid grid;
  grid.Init(10, 10, 10, 5.0f);
  size_t ch_sdf = grid.AddChannel("sdf");

  BrainSDF sdf;
  sdf.primitives.push_back({"brain", 25, 25, 25, 20, 20, 20});
  size_t ch_reg = grid.AddChannel("region");
  sdf.BakeToGrid(grid, ch_sdf, ch_reg);

  NeuralField field;
  field.ch_sdf = ch_sdf;
  field.Init(grid);

  float e_before = grid.channels[field.ch_e].data[grid.Idx(5, 5, 5)];

  // Run a few steps
  for (int i = 0; i < 10; ++i) {
    field.Step(grid, 0.5f);
  }

  float e_after = grid.channels[field.ch_e].data[grid.Idx(5, 5, 5)];

  // Activity should have evolved (not necessarily increased, but changed)
  assert(e_before != e_after);
}

TEST(neural_field_stimulus) {
  VoxelGrid grid;
  grid.Init(10, 10, 10, 5.0f);
  size_t ch_sdf = grid.AddChannel("sdf");

  BrainSDF sdf;
  sdf.primitives.push_back({"brain", 25, 25, 25, 20, 20, 20});
  size_t ch_reg = grid.AddChannel("region");
  sdf.BakeToGrid(grid, ch_sdf, ch_reg);

  NeuralField field;
  field.ch_sdf = ch_sdf;
  field.Init(grid);

  float before = field.ReadActivity(grid, 25, 25, 25);
  field.Stimulate(grid, 25, 25, 25, 5.0f, 10.0f);
  float after = field.ReadActivity(grid, 25, 25, 25);

  // Stimulus should increase activity at injection point
  assert(after > before);
}

TEST(neural_field_mask_respected) {
  VoxelGrid grid;
  grid.Init(10, 10, 10, 5.0f);
  size_t ch_sdf = grid.AddChannel("sdf");

  // Small sphere: only center region is "brain"
  BrainSDF sdf;
  sdf.primitives.push_back({"brain", 25, 25, 25, 10, 10, 10});
  size_t ch_reg = grid.AddChannel("region");
  sdf.BakeToGrid(grid, ch_sdf, ch_reg);

  NeuralField field;
  field.ch_sdf = ch_sdf;
  field.Init(grid);

  // Step many times
  for (int i = 0; i < 50; ++i) {
    field.Step(grid, 0.5f);
  }

  // Outside voxels should remain zero
  float corner_e = grid.channels[field.ch_e].data[grid.Idx(0, 0, 0)];
  float corner_i = grid.channels[field.ch_i].data[grid.Idx(0, 0, 0)];
  assert(corner_e == 0.0f);
  assert(corner_i == 0.0f);
}

// ===== LODManager tests =====

TEST(lod_manager_basic) {
  LODManager lod;
  lod.SetFocus(250, 150, 100);

  // At the focus: compartmental
  assert(lod.GetLOD(250, 150, 100) == LODLevel::kCompartmental);

  // 50um away: neuron level
  assert(lod.GetLOD(300, 150, 100) == LODLevel::kNeuron);

  // 150um away: region level
  assert(lod.GetLOD(400, 150, 100) == LODLevel::kRegion);

  // 500um away: continuum
  assert(lod.GetLOD(750, 150, 100) == LODLevel::kContinuum);
}

TEST(lod_manager_hysteresis) {
  LODManager lod;
  lod.SetFocus(0, 0, 0);
  lod.hysteresis = 10.0f;

  // Point at 95um: neuron level
  LODLevel current = LODLevel::kNeuron;
  // Move to 105um: still neuron (within hysteresis band of 100+10=110)
  LODLevel next = lod.GetLODWithHysteresis(105, 0, 0, current);
  assert(next == LODLevel::kNeuron);

  // Move to 115um: now transitions to region
  next = lod.GetLODWithHysteresis(115, 0, 0, current);
  assert(next == LODLevel::kRegion);
}

TEST(lod_manager_update_all) {
  LODManager lod;
  lod.SetFocus(0, 0, 0);

  lod.region_lods.push_back({"close", 20, 0, 0, LODLevel::kContinuum});
  lod.region_lods.push_back({"far", 500, 0, 0, LODLevel::kContinuum});

  int transitions = lod.UpdateAll();

  // "close" should escalate to compartmental
  assert(lod.region_lods[0].current_lod == LODLevel::kCompartmental);
  // "far" should stay continuum
  assert(lod.region_lods[1].current_lod == LODLevel::kContinuum);
  assert(transitions == 1);
}

// ---- Main ----

int main() {
  return RunAllTests();
}
