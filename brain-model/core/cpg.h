#ifndef FWMC_CPG_H_
#define FWMC_CPG_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include "core/neuron_array.h"

namespace fwmc {

// Central Pattern Generator: injects oscillatory current into VNC neurons
// to produce spontaneous rhythmic locomotion without keyboard input.
//
// In Drosophila, the VNC contains CPG circuits that generate tripod gait
// (3 legs swing while 3 legs stance, alternating). Descending commands
// from the brain modulate CPG frequency and amplitude, but the rhythm
// is intrinsic to the VNC (Bidaye et al. 2018, Mendes et al. 2013).
//
// This module provides tonic + oscillatory drive to VNC motor neurons,
// split into two anti-phase groups (left-forward + right-hind vs
// right-forward + left-hind) to produce tripod-like alternation.
struct CPGOscillator {
  // Two anti-phase neuron groups (tripod gait pattern)
  std::vector<uint32_t> group_a;  // L-fore + R-mid + L-hind
  std::vector<uint32_t> group_b;  // R-fore + L-mid + R-hind

  float frequency_hz = 8.0f;     // stepping frequency (Drosophila: 5-15 Hz)
  float amplitude = 6.0f;        // oscillation amplitude (current units)
  float tonic_drive = 3.0f;      // constant baseline drive (keeps neurons near threshold)
  float phase = 0.0f;            // current phase (radians)
  float drive_scale = 0.0f;      // modulation from descending commands [0,1]
                                  // 0 = CPG silent, 1 = full amplitude
  bool initialized = false;

  // Auto-assign VNC motor neurons to tripod groups.
  // Uses spatial position: neurons below midline-y go to group A,
  // above to group B (rough L/R alternation proxy).
  // vnc_region: region index for VNC (5 in drosophila_full.brain)
  // sensory_fraction: skip first N% of VNC neurons (they're sensory)
  void Init(const NeuronArray& neurons, uint8_t vnc_region,
            float midline_x = 250.0f, float sensory_fraction = 0.3f) {
    group_a.clear();
    group_b.clear();

    // Collect VNC neurons
    std::vector<uint32_t> vnc;
    for (size_t i = 0; i < neurons.n; ++i) {
      if (neurons.region[i] == vnc_region)
        vnc.push_back(static_cast<uint32_t>(i));
    }
    if (vnc.empty()) return;

    // Skip sensory neurons (first fraction, used by proprioception)
    size_t motor_start = static_cast<size_t>(vnc.size() * sensory_fraction);

    // Split remaining VNC neurons into two anti-phase groups by x-position.
    // Neurons left of midline go to group A, right to group B.
    // This creates L/R alternation that maps to tripod gait through
    // the existing motor output asymmetry decoding.
    for (size_t idx = motor_start; idx < vnc.size(); ++idx) {
      uint32_t ni = vnc[idx];
      if (neurons.x[ni] < midline_x)
        group_a.push_back(ni);
      else
        group_b.push_back(ni);
    }

    initialized = !group_a.empty() && !group_b.empty();
  }

  // Advance CPG phase and inject oscillatory current.
  // Call once per brain timestep (1ms).
  // descending_drive: modulation from brain [0,1]. 0 = CPG off, 1 = full.
  void Step(NeuronArray& neurons, float dt_ms, float descending_drive) {
    if (!initialized) return;

    // Smooth drive transitions (don't snap CPG on/off)
    float alpha = 1.0f - std::exp(-dt_ms / 50.0f);  // ~50ms time constant
    drive_scale += alpha * (descending_drive - drive_scale);

    if (drive_scale < 0.01f) return;  // CPG effectively off

    // Advance phase
    phase += 2.0f * 3.14159265f * frequency_hz * dt_ms / 1000.0f;
    if (phase > 2.0f * 3.14159265f) phase -= 2.0f * 3.14159265f;

    // Oscillatory current: group A gets sin(phase), group B gets sin(phase + pi)
    float osc_a = std::sin(phase);
    float osc_b = std::sin(phase + 3.14159265f);  // anti-phase

    // Current = tonic + oscillation * amplitude * drive
    float current_a = (tonic_drive + osc_a * amplitude) * drive_scale;
    float current_b = (tonic_drive + osc_b * amplitude) * drive_scale;

    // Only inject positive current (negative doesn't drive spiking neurons)
    current_a = std::max(0.0f, current_a);
    current_b = std::max(0.0f, current_b);

    for (uint32_t i : group_a) neurons.i_ext[i] += current_a;
    for (uint32_t i : group_b) neurons.i_ext[i] += current_b;
  }
};

}  // namespace fwmc

#endif  // FWMC_CPG_H_
