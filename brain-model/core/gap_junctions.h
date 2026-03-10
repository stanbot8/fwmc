#ifndef FWMC_GAP_JUNCTIONS_H_
#define FWMC_GAP_JUNCTIONS_H_

#include <cstdint>
#include <random>
#include <vector>

#include "core/neuron_array.h"

namespace fwmc {

// Electrical gap junction storage.
// Gap junctions pass current proportional to the voltage difference
// between two coupled neurons: I_gap = g * (V_post - V_pre).
// They are bidirectional (symmetric): neuron_a receives +I, neuron_b
// receives -I, so current flows from the higher-voltage cell to the
// lower-voltage cell.
//
// Important in Drosophila:
//   - Giant fiber system (GF1/GF2 escape response, ~1ms latency)
//   - Clock neurons (LNv synchronization via Inx6/Inx7 innexins)
//   - Antennal lobe local interneurons (fast oscillatory coupling)
struct GapJunctionTable {
  // Parallel arrays: junction j connects neuron_a[j] <-> neuron_b[j]
  // with conductance[j]. All three arrays have the same length.
  std::vector<uint32_t> neuron_a;
  std::vector<uint32_t> neuron_b;
  std::vector<float>    conductance;  // gap junction conductance (nS)

  size_t Size() const { return neuron_a.size(); }

  // Add a single gap junction between neurons a and b.
  // Only one entry is stored per pair; PropagateGapCurrents handles
  // the bidirectional current flow.
  void AddJunction(uint32_t a, uint32_t b, float g) {
    neuron_a.push_back(a);
    neuron_b.push_back(b);
    conductance.push_back(g);
  }

  // Propagate gap junction currents into neurons.i_ext.
  // For each junction j:
  //   I = g * (Vb - Va)
  //   neuron_a.i_ext += I   (current into a)
  //   neuron_b.i_ext -= I   (current into b, equal and opposite)
  //
  // Uses OpenMP when junction count exceeds 10000. Because multiple
  // junctions can target the same neuron, we use atomic adds for
  // correctness (same strategy as SynapseTable::PropagateSpikes).
  void PropagateGapCurrents(NeuronArray& neurons) const {
    const int n = static_cast<int>(Size());
    if (n == 0) return;

    float* i_ext = neurons.i_ext.data();
    const float* v = neurons.v.data();

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(n > 10000)
    #endif
    for (int j = 0; j < n; ++j) {
      uint32_t a = neuron_a[j];
      uint32_t b = neuron_b[j];
      float I = conductance[j] * (v[b] - v[a]);
      #ifdef _OPENMP
      #pragma omp atomic
      #endif
      i_ext[a] += I;
      #ifdef _OPENMP
      #pragma omp atomic
      #endif
      i_ext[b] -= I;
    }
  }

  // Connect neurons within a region with gap junctions at a given
  // probability (density). Useful for building clock neuron networks
  // or antennal lobe coupling.
  //
  // For each pair (i, j) with i < j in the specified region, a junction
  // is created with probability `density` and conductance `g_default`.
  void BuildFromRegion(const NeuronArray& neurons, uint8_t region,
                       float density, float g_default,
                       uint32_t seed = 42) {
    // Collect neuron indices in the target region
    std::vector<uint32_t> members;
    members.reserve(256);
    for (size_t i = 0; i < neurons.n; ++i) {
      if (neurons.region[i] == region) {
        members.push_back(static_cast<uint32_t>(i));
      }
    }

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> coin(0.0f, 1.0f);

    for (size_t i = 0; i < members.size(); ++i) {
      for (size_t j = i + 1; j < members.size(); ++j) {
        if (coin(rng) < density) {
          AddJunction(members[i], members[j], g_default);
        }
      }
    }
  }

  void Clear() {
    neuron_a.clear();
    neuron_b.clear();
    conductance.clear();
  }
};

}  // namespace fwmc

#endif  // FWMC_GAP_JUNCTIONS_H_
