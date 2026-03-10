#ifndef FWMC_SHORT_TERM_PLASTICITY_H_
#define FWMC_SHORT_TERM_PLASTICITY_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include "core/neuron_array.h"
#include "core/synapse_table.h"

namespace fwmc {

// Preset STP parameter factories.
// STPParams is defined in synapse_table.h. These helpers return
// biologically motivated configurations.

// Facilitating synapses: low baseline release, slow facilitation decay,
// fast depression recovery. Repeated spikes progressively increase
// transmission (e.g., cortical E-to-E, some mushroom body inputs).
inline STPParams STPFacilitating() {
  return {.U_se = 0.15f, .tau_d = 50.0f, .tau_f = 1500.0f};
}

// Depressing synapses: high baseline release, fast facilitation decay,
// slow depression recovery. First spike transmits strongly, subsequent
// spikes weaken (e.g., thalamocortical, sensory-to-interneuron).
inline STPParams STPDepressing() {
  return {.U_se = 0.5f, .tau_d = 200.0f, .tau_f = 50.0f};
}

// Combined facilitation and depression: intermediate parameters that
// produce non-monotonic response (initial facilitation then depression
// at high rates). Seen in some cortical E-to-I connections.
inline STPParams STPCombined() {
  return {.U_se = 0.25f, .tau_d = 150.0f, .tau_f = 500.0f};
}

// Update STP state for all synapses in a SynapseTable.
//
// Call this once per timestep, after spike propagation. For each synapse:
//   1. Relax u toward U_se and x toward 1.0 (exponential decay)
//   2. If pre-neuron spiked: u += U_se*(1-u), x -= u*x
//
// Requires SynapseTable::HasSTP() == true (call InitSTP first).
// OpenMP parallelized for large synapse counts.
inline void UpdateSTP(SynapseTable& synapses, const NeuronArray& neurons,
                      float dt_ms) {
  if (!synapses.HasSTP()) return;
  const size_t n_syn = synapses.Size();
  const int n_pre = static_cast<int>(synapses.n_neurons);

  // Phase 1: exponential relaxation (embarrassingly parallel)
  #ifdef _OPENMP
  #pragma omp parallel for schedule(static) if(n_syn > 50000)
  #endif
  for (int s = 0; s < static_cast<int>(n_syn); ++s) {
    float alpha_f = 1.0f - std::exp(-dt_ms / synapses.stp_tau_f[s]);
    float alpha_d = 1.0f - std::exp(-dt_ms / synapses.stp_tau_d[s]);
    synapses.stp_u[s] += (synapses.stp_U_se[s] - synapses.stp_u[s]) * alpha_f;
    synapses.stp_x[s] += (1.0f - synapses.stp_x[s]) * alpha_d;
  }

  // Phase 2: spike updates (iterate by pre-neuron for CSR locality)
  for (int pre = 0; pre < n_pre; ++pre) {
    if (!neurons.spiked[pre]) continue;
    const uint32_t start = synapses.row_ptr[pre];
    const uint32_t end   = synapses.row_ptr[pre + 1];
    for (uint32_t s = start; s < end; ++s) {
      float U = synapses.stp_U_se[s];
      synapses.stp_u[s] += U * (1.0f - synapses.stp_u[s]);
      synapses.stp_u[s] = std::clamp(synapses.stp_u[s], 0.0f, 1.0f);
      float ux = synapses.stp_u[s] * synapses.stp_x[s];
      synapses.stp_x[s] = std::max(0.0f, synapses.stp_x[s] - ux);
    }
  }
}

// Reset all STP state to resting values without reallocating.
inline void ResetSTP(SynapseTable& synapses) {
  if (!synapses.HasSTP()) return;
  for (size_t s = 0; s < synapses.Size(); ++s) {
    synapses.stp_u[s] = synapses.stp_U_se[s];
    synapses.stp_x[s] = 1.0f;
  }
}

// Diagnostic: mean utilization across all synapses.
inline float MeanSTPUtilization(const SynapseTable& synapses) {
  if (!synapses.HasSTP() || synapses.Size() == 0) return 0.0f;
  float sum = 0.0f;
  for (size_t s = 0; s < synapses.Size(); ++s) sum += synapses.stp_u[s];
  return sum / static_cast<float>(synapses.Size());
}

// Diagnostic: mean available resources across all synapses.
inline float MeanSTPResources(const SynapseTable& synapses) {
  if (!synapses.HasSTP() || synapses.Size() == 0) return 0.0f;
  float sum = 0.0f;
  for (size_t s = 0; s < synapses.Size(); ++s) sum += synapses.stp_x[s];
  return sum / static_cast<float>(synapses.Size());
}

}  // namespace fwmc

#endif  // FWMC_SHORT_TERM_PLASTICITY_H_
