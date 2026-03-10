#ifndef FWMC_CELL_TYPES_H_
#define FWMC_CELL_TYPES_H_

#include <unordered_map>
#include <vector>
#include "core/experiment_config.h"
#include "core/izhikevich.h"
#include "core/neuron_array.h"

namespace fwmc {

// Manages per-neuron Izhikevich parameters based on cell type assignments.
// Supports config-driven overrides: load defaults from ParamsForCellType(),
// then override individual params from a config map.
struct CellTypeManager {
  // Per-neuron params (indexed by neuron index)
  std::vector<IzhikevichParams> neuron_params;

  // Custom overrides: CellType -> IzhikevichParams
  std::unordered_map<uint8_t, IzhikevichParams> overrides;

  // Assign default params to all neurons based on their type field.
  void AssignFromTypes(const NeuronArray& neurons) {
    neuron_params.resize(neurons.n);
    for (size_t i = 0; i < neurons.n; ++i) {
      auto ct = static_cast<CellType>(neurons.type[i]);
      auto it = overrides.find(neurons.type[i]);
      if (it != overrides.end()) {
        neuron_params[i] = it->second;
      } else {
        neuron_params[i] = ParamsForCellType(ct);
      }
    }
  }

  // Get params for a single neuron.
  const IzhikevichParams& Get(size_t idx) const {
    return neuron_params[idx];
  }

  // Override params for a cell type. Call AssignFromTypes() after to apply.
  void SetOverride(CellType ct, const IzhikevichParams& p) {
    overrides[static_cast<uint8_t>(ct)] = p;
  }
};

// Step neurons with per-neuron params (heterogeneous dynamics).
// Slower than the uniform-param IzhikevichStep, but biologically richer.
inline void IzhikevichStepHeterogeneous(NeuronArray& neurons, float dt_ms,
                                         float sim_time_ms,
                                         const CellTypeManager& types) {
  if (types.neuron_params.size() < neurons.n) return;  // safety: params not assigned
  const int n = static_cast<int>(neurons.n);
  float* FWMC_RESTRICT v = neurons.v.data();
  float* FWMC_RESTRICT u = neurons.u.data();
  const float* FWMC_RESTRICT i_syn = neurons.i_syn.data();
  const float* FWMC_RESTRICT i_ext = neurons.i_ext.data();
  uint8_t* FWMC_RESTRICT spiked = neurons.spiked.data();
  float* FWMC_RESTRICT last_spike = neurons.last_spike_time.data();

  #ifdef _OPENMP
  #pragma omp parallel for schedule(static) if(n > 10000)
  #endif
  for (int i = 0; i < n; ++i) {
    const auto& p = types.neuron_params[static_cast<size_t>(i)];
    float vi = v[i];
    float ui = u[i];
    float I = i_syn[i] + i_ext[i];

    vi += 0.5f * dt_ms * (0.04f * vi * vi + 5.0f * vi + 140.0f - ui + I);
    vi += 0.5f * dt_ms * (0.04f * vi * vi + 5.0f * vi + 140.0f - ui + I);
    ui += dt_ms * p.a * (p.b * vi - ui);

    // Clamp runaway voltages and reset divergent neurons
    if (!std::isfinite(vi) || vi > 100.0f) vi = p.c;
    if (!std::isfinite(ui) || std::abs(ui) > 1e6f) ui = p.b * p.c;

    bool in_refractory = (sim_time_ms - last_spike[i]) < p.refractory_ms;
    uint8_t fired = (!in_refractory && vi >= p.v_thresh) ? 1 : 0;
    if (fired) {
      vi = p.c;
      ui += p.d;
      last_spike[i] = sim_time_ms;
    }

    v[i] = vi;
    u[i] = ui;
    spiked[i] = fired;
  }
}

}  // namespace fwmc

#endif  // FWMC_CELL_TYPES_H_
