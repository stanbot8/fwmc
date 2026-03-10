#ifndef FWMC_EXPERIMENT_CONFIG_H_
#define FWMC_EXPERIMENT_CONFIG_H_

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include "core/izhikevich.h"
#include "core/stimulus_event.h"

namespace fwmc {

// Brain region identifiers matching FlyWire neuropil annotations
enum class Region : uint8_t {
  kUnknown = 0,
  kAL = 1,       // antennal lobe
  kMB = 2,       // mushroom body (Kenyon cells)
  kMBON = 3,     // mushroom body output neurons
  kDAN = 4,      // dopaminergic neurons
  kLH = 5,       // lateral horn
  kCX = 6,       // central complex
  kOL = 7,       // optic lobe
  kSEZ = 8,      // subesophageal zone
  kPN = 9,       // projection neurons
};

// Functional cell type within a region
enum class CellType : uint8_t {
  kGeneric = 0,
  kKenyonCell = 1,
  kMBON_cholinergic = 2,
  kMBON_gabaergic = 3,
  kMBON_glutamatergic = 4,
  kDAN_PPL1 = 5,
  kDAN_PAM = 6,
  kPN_excitatory = 7,
  kPN_inhibitory = 8,
  kLN_local = 9,        // local interneuron
  kORN = 10,            // olfactory receptor neuron
  kFastSpiking = 11,
  kBursting = 12,
};

// Per-cell-type Izhikevich parameters from literature
inline IzhikevichParams ParamsForCellType(CellType ct) {
  switch (ct) {
    case CellType::kKenyonCell:
      return {0.02f, 0.2f, -65.0f, 8.0f, 30.0f};  // regular spiking
    case CellType::kFastSpiking:
    case CellType::kPN_inhibitory:
    case CellType::kLN_local:
      return {0.1f, 0.2f, -65.0f, 2.0f, 30.0f};   // fast spiking
    case CellType::kBursting:
    case CellType::kPN_excitatory:
      return {0.02f, 0.2f, -50.0f, 2.0f, 30.0f};  // bursting
    case CellType::kDAN_PPL1:
    case CellType::kDAN_PAM:
      return {0.02f, 0.25f, -65.0f, 6.0f, 30.0f};  // regular spiking (tonic)
    case CellType::kORN:
      return {0.02f, 0.2f, -65.0f, 8.0f, 30.0f};   // regular spiking
    default:
      return {0.02f, 0.2f, -65.0f, 8.0f, 30.0f};   // default regular spiking
  }
}

// Experiment configuration loaded from JSON or built programmatically
struct ExperimentConfig {
  // Metadata
  std::string name;
  std::string fly_strain;
  std::string date;
  std::string notes;

  // Simulation parameters
  float dt_ms = 0.1f;
  float duration_ms = 10000.0f;
  float weight_scale = 1.0f;
  int metrics_interval = 1000;
  bool enable_stdp = false;

  // Bridge mode (0=open-loop, 1=shadow, 2=closed-loop)
  int bridge_mode = 0;

  // Replacement thresholds
  float monitor_threshold = 0.6f;
  float bridge_threshold = 0.8f;
  float resync_threshold = 0.4f;
  float min_observation_ms = 10000.0f;

  // Neurons to monitor/replace (by index)
  std::vector<uint32_t> monitor_neurons;

  // Per-neuron cell type assignments (index → CellType)
  std::unordered_map<uint32_t, CellType> neuron_types;

  // Calibration
  int calibration_interval = 10000;  // apply gradient updates every N steps (0=disabled)
  float calibration_lr = 0.001f;     // learning rate for supervised calibration

  // Stimulus protocol: ordered list of timed events
  std::vector<StimulusEvent> stimulus_protocol;

  // Data paths
  std::string connectome_dir = "data";
  std::string recording_input;   // path to pre-recorded neural data (empty = none)
  std::string output_dir = "results";

  // Recording options
  bool record_spikes = true;
  bool record_voltages = false;        // expensive: records all v[i] each step
  bool record_shadow_metrics = true;
  bool record_per_neuron_error = true;
  int recording_interval = 1;          // record every N steps (1 = every step)
};

}  // namespace fwmc

#endif  // FWMC_EXPERIMENT_CONFIG_H_
