#!/usr/bin/env python3
"""Validate FWMC simulation output against published Drosophila electrophysiology data.

Compares simulation spike data against reference values from published
neuroscience literature to assess biological plausibility.

References:
    Turner et al. 2008 - PN firing rates
    Aso et al. 2014 - MB circuit cell types
    Honegger et al. 2011 - KC population sparseness
    Tanaka et al. 2009 - AL oscillation frequencies
    Mazor & Laurent 2005 - PN/KC temporal dynamics
    Lin et al. 2014 - Correlation structure in MB

Usage:
    python3 scripts/validate_electrophysiology.py results/ --report validation_report.json
"""

import argparse
import json
import os
import struct
import sys
from collections import defaultdict

import numpy as np
from scipy import signal


# ---------------------------------------------------------------------------
# CellType enum (mirrors src/core/experiment_config.h)
# ---------------------------------------------------------------------------
CELL_TYPE_GENERIC = 0
CELL_TYPE_KC = 1          # Kenyon Cell
CELL_TYPE_MBON_CHOL = 2   # MBON cholinergic
CELL_TYPE_MBON_GABA = 3   # MBON GABAergic
CELL_TYPE_MBON_GLUT = 4   # MBON glutamatergic
CELL_TYPE_DAN_PPL1 = 5    # DAN PPL1
CELL_TYPE_DAN_PAM = 6     # DAN PAM
CELL_TYPE_PN_EXC = 7      # Projection neuron excitatory
CELL_TYPE_PN_INH = 8      # Projection neuron inhibitory
CELL_TYPE_LN = 9          # Local interneuron
CELL_TYPE_ORN = 10         # Olfactory receptor neuron
CELL_TYPE_FAST_SPIKING = 11
CELL_TYPE_BURSTING = 12

# Groups for validation
CELL_GROUP_KC = {CELL_TYPE_KC}
CELL_GROUP_MBON = {CELL_TYPE_MBON_CHOL, CELL_TYPE_MBON_GABA, CELL_TYPE_MBON_GLUT}
CELL_GROUP_DAN = {CELL_TYPE_DAN_PPL1, CELL_TYPE_DAN_PAM}
CELL_GROUP_PN = {CELL_TYPE_PN_EXC, CELL_TYPE_PN_INH}
CELL_GROUP_LN = {CELL_TYPE_LN}

CELL_GROUP_NAMES = {
    "KC": CELL_GROUP_KC,
    "MBON": CELL_GROUP_MBON,
    "DAN": CELL_GROUP_DAN,
    "PN": CELL_GROUP_PN,
    "LN": CELL_GROUP_LN,
}

# Approximate region mapping: KCs and MBONs are in MB, PNs and LNs are in AL
REGION_MB = CELL_GROUP_KC | CELL_GROUP_MBON | CELL_GROUP_DAN
REGION_AL = CELL_GROUP_PN | CELL_GROUP_LN | {CELL_TYPE_ORN}


# ---------------------------------------------------------------------------
# Reference data from published literature (hardcoded)
# ---------------------------------------------------------------------------
REFERENCE_FIRING_RATES = {
    # (min_hz, max_hz, mean_hz, std_hz, citation)
    "KC":   (0.5, 5.0, 2.0, 1.5, "Turner et al. 2008; Aso et al. 2014"),
    "PN":   (5.0, 30.0, 15.0, 8.0, "Turner et al. 2008"),
    "LN":   (10.0, 50.0, 25.0, 12.0, "Chou et al. 2010"),
    "MBON": (2.0, 15.0, 7.0, 4.0, "Aso et al. 2014"),
    "DAN":  (1.0, 10.0, 5.0, 3.0, "Aso et al. 2014"),
}

REFERENCE_SPARSENESS = {
    # (max_fraction_active, mean, std, citation)
    "MB_KC_active_frac":  (0.10, 0.05, 0.03, "Honegger et al. 2011"),
    "AL_PN_active_frac":  (0.60, 0.45, 0.10, "Honegger et al. 2011"),
}

REFERENCE_OSCILLATIONS = {
    # (min_hz, max_hz, mean_peak_hz, std, citation)
    "AL_LFP_peak":  (10.0, 30.0, 20.0, 5.0, "Tanaka et al. 2009"),
}

REFERENCE_TEMPORAL = {
    # adaptation_index: ratio of late-to-early firing (< 1.0 means adaptation)
    "PN_adaptation_index": (0.3, 0.8, 0.55, 0.15, "Mazor & Laurent 2005"),
    # KC transience: fraction of response in first 200ms window
    "KC_transience":       (0.6, 1.0, 0.8, 0.1, "Mazor & Laurent 2005"),
}

REFERENCE_CORRELATIONS = {
    # (min, max, mean, std, citation)
    "KC_KC_corr":     (-0.05, 0.15, 0.05, 0.03, "Lin et al. 2014"),
    "PN_PN_corr":     (0.05, 0.35, 0.20, 0.08, "Lin et al. 2014"),
    "within_gt_between": (True, None, None, None, "Lin et al. 2014"),
}


# ---------------------------------------------------------------------------
# Binary I/O (matches recorder.h and connectome_export.h)
# ---------------------------------------------------------------------------
def read_spikes(path):
    """Read spikes.bin -> (n_neurons, n_steps, times, spike_matrix).

    Format: [n_neurons:u32, n_steps:u32] then per step [time:f32, spiked:u8*n_neurons]
    Returns spike_matrix as (n_steps, n_neurons) uint8 array.
    """
    with open(path, "rb") as f:
        n_neurons, n_steps = struct.unpack("<II", f.read(8))
        times = []
        spikes = []
        for _ in range(n_steps):
            t = struct.unpack("<f", f.read(4))[0]
            times.append(t)
            frame = np.frombuffer(f.read(n_neurons), dtype=np.uint8)
            spikes.append(frame.copy())
    return n_neurons, n_steps, np.array(times), np.array(spikes)


def read_neurons(path):
    """Read neurons.bin -> (n, root_ids, x, y, z, types).

    Format: [count:u32] then per neuron [root_id:u64, x:f32, y:f32, z:f32, type:u8]
    """
    with open(path, "rb") as f:
        count = struct.unpack("<I", f.read(4))[0]
        root_ids = np.zeros(count, dtype=np.uint64)
        x = np.zeros(count, dtype=np.float32)
        y = np.zeros(count, dtype=np.float32)
        z = np.zeros(count, dtype=np.float32)
        types = np.zeros(count, dtype=np.uint8)
        record_size = 8 + 4 * 3 + 1  # u64 + 3*f32 + u8 = 21 bytes
        for i in range(count):
            data = f.read(record_size)
            rid, xi, yi, zi, ti = struct.unpack("<Q3fB", data)
            root_ids[i] = rid
            x[i] = xi
            y[i] = yi
            z[i] = zi
            types[i] = ti
    return count, root_ids, x, y, z, types


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------
def compute_firing_rates(spikes, times, neuron_types):
    """Compute per-cell-group mean and std firing rates in Hz.

    Args:
        spikes: (n_steps, n_neurons) uint8 array
        times: (n_steps,) float array in ms
        neuron_types: (n_neurons,) uint8 array of CellType values

    Returns:
        dict mapping group name -> {"mean_hz", "std_hz", "per_neuron_hz"}
    """
    duration_s = (times[-1] - times[0]) / 1000.0
    if duration_s <= 0:
        return {}

    per_neuron_counts = spikes.sum(axis=0).astype(np.float64)
    per_neuron_hz = per_neuron_counts / duration_s

    results = {}
    for name, type_set in CELL_GROUP_NAMES.items():
        mask = np.isin(neuron_types, list(type_set))
        if mask.sum() == 0:
            continue
        rates = per_neuron_hz[mask]
        results[name] = {
            "mean_hz": float(np.mean(rates)),
            "std_hz": float(np.std(rates)),
            "n_neurons": int(mask.sum()),
            "per_neuron_hz": rates,
        }
    return results


def compute_sparseness(spikes, neuron_types):
    """Compute population sparseness per region/cell group.

    Population sparseness = fraction of neurons active in a given time window.
    We compute per-step fraction active, then average.

    Returns:
        dict with "MB_KC_active_frac" and "AL_PN_active_frac"
    """
    results = {}

    # KC sparseness (MB region)
    kc_mask = np.isin(neuron_types, list(CELL_GROUP_KC))
    n_kc = kc_mask.sum()
    if n_kc > 0:
        kc_spikes = spikes[:, kc_mask]
        frac_active_per_step = kc_spikes.sum(axis=1) / n_kc
        results["MB_KC_active_frac"] = {
            "mean": float(np.mean(frac_active_per_step)),
            "std": float(np.std(frac_active_per_step)),
            "n_neurons": int(n_kc),
        }

    # PN sparseness (AL region)
    pn_mask = np.isin(neuron_types, list(CELL_GROUP_PN))
    n_pn = pn_mask.sum()
    if n_pn > 0:
        pn_spikes = spikes[:, pn_mask]
        frac_active_per_step = pn_spikes.sum(axis=1) / n_pn
        results["AL_PN_active_frac"] = {
            "mean": float(np.mean(frac_active_per_step)),
            "std": float(np.std(frac_active_per_step)),
            "n_neurons": int(n_pn),
        }

    return results


def compute_oscillations(spikes, times, neuron_types):
    """Compute power spectrum of population activity to detect oscillations.

    Uses FFT on the population spike count time series for AL neurons.
    Reference: Tanaka et al. 2009 - AL LFP shows 10-30 Hz oscillations.

    Returns:
        dict with "AL_LFP_peak" containing peak frequency and power spectrum info.
    """
    results = {}

    # AL population activity (PNs + LNs + ORNs)
    al_mask = np.isin(neuron_types, list(REGION_AL))
    n_al = al_mask.sum()
    if n_al < 2:
        return results

    al_counts = spikes[:, al_mask].sum(axis=1).astype(np.float64)

    # Estimate sampling rate from time array
    dt_ms = np.median(np.diff(times))
    if dt_ms <= 0:
        return results
    fs = 1000.0 / dt_ms  # Hz

    if len(al_counts) < 64:
        return results

    # Compute power spectral density using Welch's method
    nperseg = min(256, len(al_counts))
    freqs, psd = signal.welch(al_counts, fs=fs, nperseg=nperseg)

    # Find peak in 5-100 Hz range (biological oscillation band)
    band_mask = (freqs >= 5.0) & (freqs <= 100.0)
    if not band_mask.any():
        return results

    band_freqs = freqs[band_mask]
    band_psd = psd[band_mask]
    peak_idx = np.argmax(band_psd)
    peak_freq = float(band_freqs[peak_idx])
    peak_power = float(band_psd[peak_idx])

    # Check for MB oscillations (should be weak/absent)
    mb_mask = np.isin(neuron_types, list(REGION_MB))
    n_mb = mb_mask.sum()
    mb_peak_power = 0.0
    if n_mb > 2 and len(spikes[:, mb_mask].sum(axis=1)) >= 64:
        mb_counts = spikes[:, mb_mask].sum(axis=1).astype(np.float64)
        mb_freqs, mb_psd = signal.welch(mb_counts, fs=fs, nperseg=nperseg)
        mb_band = (mb_freqs >= 5.0) & (mb_freqs <= 100.0)
        if mb_band.any():
            mb_peak_power = float(np.max(mb_psd[mb_band]))

    results["AL_LFP_peak"] = {
        "peak_freq_hz": peak_freq,
        "peak_power": peak_power,
        "mb_peak_power": mb_peak_power,
        "al_stronger_than_mb": peak_power > mb_peak_power * 2.0,
    }

    return results


def compute_temporal_dynamics(spikes, times, neuron_types, stim_onset_ms=None):
    """Compute PSTH and adaptation index for PNs and KCs.

    Reference:
        Mazor & Laurent 2005: PN adaptation over 500ms, KC transience < 200ms.

    Args:
        stim_onset_ms: stimulus onset time. If None, uses first time point.

    Returns:
        dict with "PN_adaptation_index" and "KC_transience".
    """
    results = {}
    if stim_onset_ms is None:
        stim_onset_ms = float(times[0])

    duration_ms = float(times[-1] - stim_onset_ms)

    # PN adaptation: compare early (0-250ms) vs late (250-500ms) firing
    pn_mask = np.isin(neuron_types, list(CELL_GROUP_PN))
    n_pn = pn_mask.sum()
    if n_pn > 0 and duration_ms >= 500.0:
        pn_spikes = spikes[:, pn_mask]
        early_mask = (times >= stim_onset_ms) & (times < stim_onset_ms + 250.0)
        late_mask = (times >= stim_onset_ms + 250.0) & (times < stim_onset_ms + 500.0)

        early_rate = pn_spikes[early_mask].sum() / max(early_mask.sum(), 1)
        late_rate = pn_spikes[late_mask].sum() / max(late_mask.sum(), 1)

        adaptation_index = float(late_rate / max(early_rate, 1e-9))
        results["PN_adaptation_index"] = {
            "value": adaptation_index,
            "early_rate": float(early_rate),
            "late_rate": float(late_rate),
            "n_neurons": int(n_pn),
        }

    # KC transience: fraction of spikes in first 200ms vs total response
    kc_mask = np.isin(neuron_types, list(CELL_GROUP_KC))
    n_kc = kc_mask.sum()
    if n_kc > 0 and duration_ms >= 400.0:
        kc_spikes = spikes[:, kc_mask]
        first_200_mask = (times >= stim_onset_ms) & (times < stim_onset_ms + 200.0)
        full_mask = (times >= stim_onset_ms) & (times < stim_onset_ms + 400.0)

        spikes_first_200 = kc_spikes[first_200_mask].sum()
        spikes_total = kc_spikes[full_mask].sum()

        transience = float(spikes_first_200 / max(spikes_total, 1))
        results["KC_transience"] = {
            "value": transience,
            "spikes_first_200ms": int(spikes_first_200),
            "spikes_total_400ms": int(spikes_total),
            "n_neurons": int(n_kc),
        }

    return results


def compute_correlations(spikes, neuron_types):
    """Compute within- and between-region pairwise spike correlations.

    Reference: Lin et al. 2014 - KC-KC ~0.05, PN-PN ~0.2,
    within-region > between-region.

    Uses binned spike counts and samples neuron pairs for efficiency.

    Returns:
        dict with correlation metrics.
    """
    results = {}
    max_sample = 200  # max neurons to sample per group for correlation

    def _sample_corr(spike_matrix, n_sample):
        """Compute mean pairwise correlation from a (n_steps, n_neurons) matrix."""
        n = spike_matrix.shape[1]
        if n < 2:
            return np.nan
        if n > n_sample:
            idx = np.random.choice(n, n_sample, replace=False)
            spike_matrix = spike_matrix[:, idx]
        counts = spike_matrix.astype(np.float64)
        # Bin into 10ms windows for smoother correlations
        bin_size = max(1, counts.shape[0] // max(counts.shape[0] // 10, 1))
        if bin_size > 1 and counts.shape[0] >= bin_size * 2:
            n_bins = counts.shape[0] // bin_size
            counts = counts[:n_bins * bin_size].reshape(n_bins, bin_size, -1).sum(axis=1)
        # Remove neurons with zero variance
        stds = counts.std(axis=0)
        valid = stds > 0
        if valid.sum() < 2:
            return 0.0
        counts = counts[:, valid]
        cc = np.corrcoef(counts.T)
        # Extract upper triangle (exclude diagonal)
        triu_idx = np.triu_indices(cc.shape[0], k=1)
        return float(np.mean(cc[triu_idx]))

    np.random.seed(42)  # reproducible sampling

    # KC-KC correlations
    kc_mask = np.isin(neuron_types, list(CELL_GROUP_KC))
    if kc_mask.sum() >= 2:
        kc_corr = _sample_corr(spikes[:, kc_mask], max_sample)
        results["KC_KC_corr"] = {"value": kc_corr, "n_neurons": int(kc_mask.sum())}

    # PN-PN correlations
    pn_mask = np.isin(neuron_types, list(CELL_GROUP_PN))
    if pn_mask.sum() >= 2:
        pn_corr = _sample_corr(spikes[:, pn_mask], max_sample)
        results["PN_PN_corr"] = {"value": pn_corr, "n_neurons": int(pn_mask.sum())}

    # Within-region vs between-region
    mb_mask = np.isin(neuron_types, list(REGION_MB))
    al_mask = np.isin(neuron_types, list(REGION_AL))

    within_mb = _sample_corr(spikes[:, mb_mask], max_sample) if mb_mask.sum() >= 2 else np.nan
    within_al = _sample_corr(spikes[:, al_mask], max_sample) if al_mask.sum() >= 2 else np.nan

    # Between-region: sample from both and compute cross-correlation
    n_mb = mb_mask.sum()
    n_al = al_mask.sum()
    between_corr = np.nan
    if n_mb >= 2 and n_al >= 2:
        n_sample = min(max_sample // 2, n_mb, n_al)
        mb_idx = np.random.choice(n_mb, n_sample, replace=False)
        al_idx = np.random.choice(n_al, n_sample, replace=False)
        combined = np.hstack([
            spikes[:, mb_mask][:, mb_idx],
            spikes[:, al_mask][:, al_idx],
        ]).astype(np.float64)
        if combined.shape[0] > 10:
            cc = np.corrcoef(combined.T)
            # Cross-region block: top-right
            cross = cc[:n_sample, n_sample:]
            between_corr = float(np.nanmean(cross))

    within_mean = np.nanmean([x for x in [within_mb, within_al] if np.isfinite(x)])
    results["within_gt_between"] = {
        "within_mb": float(within_mb) if np.isfinite(within_mb) else None,
        "within_al": float(within_al) if np.isfinite(within_al) else None,
        "between": float(between_corr) if np.isfinite(between_corr) else None,
        "within_mean": float(within_mean) if np.isfinite(within_mean) else None,
        "pass": bool(np.isfinite(within_mean) and np.isfinite(between_corr)
                      and within_mean > between_corr),
    }

    return results


def compute_cv_isi(spikes, times):
    """Compute coefficient of variation of inter-spike intervals.

    For Poisson-like spiking, CV_ISI ~ 1.0.
    Regular spiking has CV_ISI < 1; bursting has CV_ISI > 1.

    Returns:
        dict with "mean_cv", "std_cv", "per_neuron_cv".
    """
    n_neurons = spikes.shape[1]
    cvs = []
    for i in range(n_neurons):
        spike_times = times[spikes[:, i] > 0]
        if len(spike_times) < 3:
            continue
        isis = np.diff(spike_times)
        if isis.mean() > 0:
            cvs.append(float(isis.std() / isis.mean()))
    if not cvs:
        return {"mean_cv": np.nan, "std_cv": np.nan, "per_neuron_cv": []}
    return {
        "mean_cv": float(np.mean(cvs)),
        "std_cv": float(np.std(cvs)),
        "per_neuron_cv": cvs,
    }


def compute_fano_factor(spikes, times, window_ms=50.0):
    """Compute Fano factor (variance / mean of spike counts in windows).

    For Poisson process, Fano factor ~ 1.0.

    Args:
        window_ms: window size for counting spikes.

    Returns:
        dict with "mean_fano", "std_fano", "per_neuron_fano".
    """
    dt_ms = np.median(np.diff(times))
    if dt_ms <= 0:
        return {"mean_fano": np.nan, "std_fano": np.nan}

    steps_per_window = max(1, int(window_ms / dt_ms))
    n_steps, n_neurons = spikes.shape
    n_windows = n_steps // steps_per_window
    if n_windows < 2:
        return {"mean_fano": np.nan, "std_fano": np.nan}

    # Reshape into windows and count spikes per window per neuron
    truncated = spikes[:n_windows * steps_per_window]
    windowed = truncated.reshape(n_windows, steps_per_window, n_neurons)
    counts = windowed.sum(axis=1).astype(np.float64)  # (n_windows, n_neurons)

    means = counts.mean(axis=0)
    variances = counts.var(axis=0)

    # Only include neurons with nonzero mean
    active = means > 0
    if active.sum() == 0:
        return {"mean_fano": np.nan, "std_fano": np.nan}

    fano = variances[active] / means[active]
    return {
        "mean_fano": float(np.mean(fano)),
        "std_fano": float(np.std(fano)),
        "n_active_neurons": int(active.sum()),
    }


# ---------------------------------------------------------------------------
# Validation scoring
# ---------------------------------------------------------------------------
def _score_range(value, ref_min, ref_max, ref_mean, ref_std):
    """Score a value against a reference range.

    Returns (match_score, z_score, passed).
    """
    if ref_std is not None and ref_std > 0:
        z = abs(value - ref_mean) / ref_std
    else:
        z = 0.0

    # Match score: 1.0 if within reference range, decays outside
    if ref_min <= value <= ref_max:
        match = 1.0
    else:
        dist = min(abs(value - ref_min), abs(value - ref_max))
        span = ref_max - ref_min
        if span > 0:
            match = max(0.0, 1.0 - dist / span)
        else:
            match = max(0.0, 1.0 - abs(z) / 4.0)

    passed = bool(ref_min <= value <= ref_max) or (z < 2.0)
    return float(match), float(z), passed


def validate_firing_rates(rate_results):
    """Validate firing rates against literature references."""
    validations = []
    for group_name, ref in REFERENCE_FIRING_RATES.items():
        ref_min, ref_max, ref_mean, ref_std, citation = ref
        if group_name not in rate_results:
            validations.append({
                "metric": f"firing_rate_{group_name}",
                "status": "skipped",
                "reason": f"No {group_name} neurons found",
                "citation": citation,
            })
            continue
        obs_mean = rate_results[group_name]["mean_hz"]
        match, z, passed = _score_range(obs_mean, ref_min, ref_max, ref_mean, ref_std)
        validations.append({
            "metric": f"firing_rate_{group_name}",
            "observed_mean_hz": round(obs_mean, 3),
            "observed_std_hz": round(rate_results[group_name]["std_hz"], 3),
            "reference_range_hz": [ref_min, ref_max],
            "reference_mean_hz": ref_mean,
            "match_score": round(match, 4),
            "z_score": round(z, 4),
            "pass": passed,
            "n_neurons": rate_results[group_name]["n_neurons"],
            "citation": citation,
        })
    return validations


def validate_sparseness(sparse_results):
    """Validate population sparseness against literature references."""
    validations = []
    for key, ref in REFERENCE_SPARSENESS.items():
        ref_max_frac, ref_mean, ref_std, citation = ref
        if key not in sparse_results:
            validations.append({
                "metric": f"sparseness_{key}",
                "status": "skipped",
                "reason": f"No data for {key}",
                "citation": citation,
            })
            continue
        obs = sparse_results[key]["mean"]
        # For KC: should be < 10% active; for PN: 30-60%
        if key == "MB_KC_active_frac":
            match, z, passed = _score_range(obs, 0.0, ref_max_frac, ref_mean, ref_std)
        else:  # AL_PN_active_frac
            match, z, passed = _score_range(obs, 0.30, ref_max_frac, ref_mean, ref_std)
        validations.append({
            "metric": f"sparseness_{key}",
            "observed_frac": round(obs, 4),
            "reference_range": [0.0 if "KC" in key else 0.30, ref_max_frac],
            "match_score": round(match, 4),
            "z_score": round(z, 4),
            "pass": passed,
            "n_neurons": sparse_results[key]["n_neurons"],
            "citation": citation,
        })
    return validations


def validate_oscillations(osc_results):
    """Validate oscillation frequencies against literature references."""
    validations = []
    ref = REFERENCE_OSCILLATIONS["AL_LFP_peak"]
    ref_min, ref_max, ref_mean, ref_std, citation = ref
    if "AL_LFP_peak" not in osc_results:
        validations.append({
            "metric": "oscillation_AL_LFP",
            "status": "skipped",
            "reason": "Insufficient AL data for spectral analysis",
            "citation": citation,
        })
        return validations

    obs_freq = osc_results["AL_LFP_peak"]["peak_freq_hz"]
    match, z, passed = _score_range(obs_freq, ref_min, ref_max, ref_mean, ref_std)
    validations.append({
        "metric": "oscillation_AL_LFP",
        "observed_peak_hz": round(obs_freq, 2),
        "reference_range_hz": [ref_min, ref_max],
        "match_score": round(match, 4),
        "z_score": round(z, 4),
        "pass": passed,
        "al_stronger_than_mb": osc_results["AL_LFP_peak"].get("al_stronger_than_mb"),
        "citation": citation,
    })
    return validations


def validate_temporal(temporal_results):
    """Validate temporal dynamics against literature references."""
    validations = []
    for key, ref in REFERENCE_TEMPORAL.items():
        ref_min, ref_max, ref_mean, ref_std, citation = ref
        if key not in temporal_results:
            validations.append({
                "metric": f"temporal_{key}",
                "status": "skipped",
                "reason": f"Insufficient data for {key}",
                "citation": citation,
            })
            continue
        obs = temporal_results[key]["value"]
        match, z, passed = _score_range(obs, ref_min, ref_max, ref_mean, ref_std)
        validations.append({
            "metric": f"temporal_{key}",
            "observed": round(obs, 4),
            "reference_range": [ref_min, ref_max],
            "match_score": round(match, 4),
            "z_score": round(z, 4),
            "pass": passed,
            "citation": citation,
        })
    return validations


def validate_correlations(corr_results):
    """Validate correlation structure against literature references."""
    validations = []

    for key in ["KC_KC_corr", "PN_PN_corr"]:
        ref = REFERENCE_CORRELATIONS[key]
        ref_min, ref_max, ref_mean, ref_std, citation = ref
        if key not in corr_results:
            validations.append({
                "metric": f"correlation_{key}",
                "status": "skipped",
                "reason": f"Insufficient data for {key}",
                "citation": citation,
            })
            continue
        obs = corr_results[key]["value"]
        match, z, passed = _score_range(obs, ref_min, ref_max, ref_mean, ref_std)
        validations.append({
            "metric": f"correlation_{key}",
            "observed": round(obs, 4),
            "reference_range": [ref_min, ref_max],
            "match_score": round(match, 4),
            "z_score": round(z, 4),
            "pass": passed,
            "n_neurons": corr_results[key]["n_neurons"],
            "citation": citation,
        })

    # Within > between check
    if "within_gt_between" in corr_results:
        wgb = corr_results["within_gt_between"]
        passed = wgb.get("pass", False)
        validations.append({
            "metric": "correlation_within_gt_between",
            "within_mean": wgb.get("within_mean"),
            "between_mean": wgb.get("between"),
            "match_score": 1.0 if passed else 0.0,
            "z_score": 0.0,
            "pass": passed,
            "citation": REFERENCE_CORRELATIONS["within_gt_between"][4],
        })

    return validations


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def print_report(all_validations, cv_isi, fano_factor):
    """Print a formatted validation report to stdout."""
    print("\n" + "=" * 78)
    print("  FWMC Electrophysiology Validation Report")
    print("=" * 78)

    # Header
    print(f"\n{'Metric':<40} {'Observed':>10} {'Range':>16} {'Score':>6} {'Pass':>5}")
    print("-" * 78)

    n_scored = 0
    total_score = 0.0
    n_pass = 0
    n_fail = 0
    n_skip = 0
    flagged = []

    for v in all_validations:
        if v.get("status") == "skipped":
            n_skip += 1
            print(f"  {v['metric']:<38} {'--':>10} {'--':>16} {'SKIP':>6} {'--':>5}")
            continue

        obs_str = "--"
        for field in ["observed_mean_hz", "observed_frac", "observed_peak_hz", "observed"]:
            if field in v:
                obs_str = f"{v[field]:.3f}"
                break
        if "within_mean" in v and v.get("within_mean") is not None:
            obs_str = f"{v['within_mean']:.3f}"

        range_str = "--"
        for field in ["reference_range_hz", "reference_range"]:
            if field in v:
                r = v[field]
                range_str = f"[{r[0]:.1f},{r[1]:.1f}]"
                break

        score = v.get("match_score", 0.0)
        passed = v.get("pass", False)
        n_scored += 1
        total_score += score
        if passed:
            n_pass += 1
        else:
            n_fail += 1
            flagged.append(v["metric"])

        pass_str = "OK" if passed else "FAIL"
        print(f"  {v['metric']:<38} {obs_str:>10} {range_str:>16} {score:>6.3f} {pass_str:>5}")

    # Supplementary metrics (not scored against literature)
    print("-" * 78)
    print("  Supplementary metrics (informational):")
    if not np.isnan(cv_isi.get("mean_cv", np.nan)):
        cv_val = cv_isi["mean_cv"]
        cv_note = "Poisson-like" if 0.7 <= cv_val <= 1.3 else (
            "regular" if cv_val < 0.7 else "bursty")
        print(f"    CV(ISI):      {cv_val:.3f} +/- {cv_isi['std_cv']:.3f}  ({cv_note})")
    if not np.isnan(fano_factor.get("mean_fano", np.nan)):
        ff_val = fano_factor["mean_fano"]
        ff_note = "Poisson-like" if 0.7 <= ff_val <= 1.3 else (
            "sub-Poisson" if ff_val < 0.7 else "super-Poisson")
        print(f"    Fano factor:  {ff_val:.3f} +/- {fano_factor['std_fano']:.3f}  ({ff_note})")

    # Summary
    print("\n" + "=" * 78)
    overall = total_score / max(n_scored, 1)
    print(f"  Biological plausibility score: {overall:.3f}  "
          f"({n_pass} pass / {n_fail} fail / {n_skip} skip out of "
          f"{n_scored + n_skip} metrics)")

    if flagged:
        print(f"\n  Non-biological flags:")
        for f in flagged:
            print(f"    - {f}")
    elif n_scored > 0:
        print(f"\n  All scored metrics within biologically plausible range.")
    print("=" * 78 + "\n")

    return overall


def build_report_json(all_validations, cv_isi, fano_factor, overall_score):
    """Build the full JSON report dict."""
    return {
        "overall_biological_plausibility_score": round(overall_score, 4),
        "n_metrics_scored": sum(1 for v in all_validations if v.get("status") != "skipped"),
        "n_pass": sum(1 for v in all_validations if v.get("pass") is True),
        "n_fail": sum(1 for v in all_validations
                      if v.get("pass") is False and v.get("status") != "skipped"),
        "n_skip": sum(1 for v in all_validations if v.get("status") == "skipped"),
        "validations": all_validations,
        "supplementary": {
            "cv_isi": {k: v for k, v in cv_isi.items() if k != "per_neuron_cv"},
            "fano_factor": fano_factor,
        },
        "flagged_non_biological": [
            v["metric"] for v in all_validations
            if v.get("pass") is False and v.get("status") != "skipped"
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Validate FWMC simulation output against published "
                    "Drosophila electrophysiology data.")
    parser.add_argument("result_dir",
                        help="Path to results directory containing spikes.bin "
                             "and optionally neurons.bin")
    parser.add_argument("--report", default=None,
                        help="Path to save JSON validation report")
    parser.add_argument("--neurons", default=None,
                        help="Path to neurons.bin (default: <result_dir>/neurons.bin "
                             "or data/neurons.bin)")
    parser.add_argument("--stim-onset", type=float, default=None,
                        help="Stimulus onset time in ms (default: auto-detect)")
    args = parser.parse_args()

    if not os.path.isdir(args.result_dir):
        print(f"Error: {args.result_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # --- Load spikes ---
    spike_path = os.path.join(args.result_dir, "spikes.bin")
    if not os.path.exists(spike_path):
        print(f"Error: {spike_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Loading spikes from {spike_path} ...")
    n_neurons, n_steps, times, spikes = read_spikes(spike_path)
    print(f"  {n_neurons} neurons, {n_steps} steps, "
          f"time range {times[0]:.1f}-{times[-1]:.1f} ms")

    if n_steps < 2:
        print("Error: need at least 2 time steps for analysis", file=sys.stderr)
        sys.exit(1)

    # --- Load neuron types ---
    neuron_types = None
    neuron_candidates = []
    if args.neurons:
        neuron_candidates.append(args.neurons)
    neuron_candidates.append(os.path.join(args.result_dir, "neurons.bin"))
    # Also look in common data directories relative to result_dir
    neuron_candidates.append(os.path.join(os.path.dirname(args.result_dir.rstrip("/")),
                                          "data", "neurons.bin"))
    neuron_candidates.append(os.path.join("data", "neurons.bin"))

    for npath in neuron_candidates:
        if os.path.exists(npath):
            print(f"Loading neuron metadata from {npath} ...")
            n_loaded, _, _, _, _, neuron_types = read_neurons(npath)
            if n_loaded != n_neurons:
                print(f"  Warning: neurons.bin has {n_loaded} neurons but "
                      f"spikes.bin has {n_neurons}. Using min.", file=sys.stderr)
                min_n = min(n_loaded, n_neurons)
                neuron_types = neuron_types[:min_n]
                spikes = spikes[:, :min_n]
                n_neurons = min_n
            break

    if neuron_types is None:
        print("Warning: No neurons.bin found. Treating all neurons as generic. "
              "Cell-type-specific validation will be skipped.", file=sys.stderr)
        neuron_types = np.zeros(n_neurons, dtype=np.uint8)

    # Print cell type distribution
    unique, counts = np.unique(neuron_types, return_counts=True)
    print("  Cell type distribution:")
    type_names = {0: "Generic", 1: "KC", 2: "MBON_chol", 3: "MBON_gaba",
                  4: "MBON_glut", 5: "DAN_PPL1", 6: "DAN_PAM", 7: "PN_exc",
                  8: "PN_inh", 9: "LN", 10: "ORN", 11: "FastSpk", 12: "Burst"}
    for t, c in zip(unique, counts):
        print(f"    {type_names.get(t, f'type_{t}')}: {c}")

    # --- Run analyses ---
    print("\nComputing firing rates ...")
    rate_results = compute_firing_rates(spikes, times, neuron_types)
    for name, r in rate_results.items():
        print(f"  {name}: {r['mean_hz']:.2f} +/- {r['std_hz']:.2f} Hz "
              f"(n={r['n_neurons']})")

    print("Computing population sparseness ...")
    sparse_results = compute_sparseness(spikes, neuron_types)
    for name, r in sparse_results.items():
        print(f"  {name}: {r['mean']:.4f} +/- {r['std']:.4f}")

    print("Computing oscillations (FFT) ...")
    osc_results = compute_oscillations(spikes, times, neuron_types)
    if "AL_LFP_peak" in osc_results:
        print(f"  AL peak: {osc_results['AL_LFP_peak']['peak_freq_hz']:.1f} Hz")

    print("Computing temporal dynamics ...")
    temporal_results = compute_temporal_dynamics(spikes, times, neuron_types,
                                                 stim_onset_ms=args.stim_onset)
    for name, r in temporal_results.items():
        print(f"  {name}: {r['value']:.4f}")

    print("Computing correlations ...")
    corr_results = compute_correlations(spikes, neuron_types)
    for name, r in corr_results.items():
        if "value" in r:
            print(f"  {name}: {r['value']:.4f}")

    print("Computing CV(ISI) ...")
    cv_isi = compute_cv_isi(spikes, times)
    if not np.isnan(cv_isi["mean_cv"]):
        print(f"  mean CV(ISI): {cv_isi['mean_cv']:.3f} +/- {cv_isi['std_cv']:.3f}")

    print("Computing Fano factor ...")
    fano = compute_fano_factor(spikes, times)
    if not np.isnan(fano["mean_fano"]):
        print(f"  mean Fano: {fano['mean_fano']:.3f} +/- {fano['std_fano']:.3f}")

    # --- Validate ---
    all_validations = []
    all_validations.extend(validate_firing_rates(rate_results))
    all_validations.extend(validate_sparseness(sparse_results))
    all_validations.extend(validate_oscillations(osc_results))
    all_validations.extend(validate_temporal(temporal_results))
    all_validations.extend(validate_correlations(corr_results))

    overall = print_report(all_validations, cv_isi, fano)

    # --- Save JSON report ---
    if args.report:
        report = build_report_json(all_validations, cv_isi, fano, overall)
        report_path = args.report
        os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Detailed report saved to: {report_path}")

    # Exit code: nonzero if any metric failed
    n_fail = sum(1 for v in all_validations
                 if v.get("pass") is False and v.get("status") != "skipped")
    sys.exit(1 if n_fail > 0 else 0)


if __name__ == "__main__":
    main()
