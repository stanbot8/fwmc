#!/usr/bin/env python3
"""Single entry-point validation dashboard for FWMC simulation output.

Loads a results directory, runs all applicable validators from
``literature.lib``, prints a summary table, and generates per-module
PNG plots plus a combined ``validation_dashboard.png``.

Adapted from skibidy's validate_all.py pattern.

Usage:
    python3 literature/validate_all.py results/phase1
    python3 literature/validate_all.py results/phase1 --no-plots
    python3 literature/validate_all.py results/phase1 --out-dir figures/
"""

import argparse
import math
import os
import struct
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so ``literature.lib`` resolves
# when invoked as ``python3 literature/validate_all.py``.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from literature.lib import (  # noqa: E402
    SIM_COLOR,
    REF_COLOR,
    load_csv,
    plots_dir,
    detect_modules,
    validate_firing_rates,
    validate_sparseness,
    validate_isi,
    validate_correlation,
    plot_firing_rate_panel,
    plot_sparseness_panel,
    plot_isi_panel,
    plot_correlation_panel,
    print_summary,
)


# ===================================================================
# Source integrity check
# ===================================================================

def check_sources() -> bool:
    """Verify that required reference data files exist.

    Returns True if all sources are present, False otherwise.
    """
    from literature.lib import _DATA_DIR

    required = [
        "firing_rates_by_cell_type.csv",
        "SOURCES.yaml",
    ]
    ok = True
    for fname in required:
        p = _DATA_DIR / fname
        if not p.exists():
            print(f"[WARN] Missing reference file: {p}", file=sys.stderr)
            ok = False
    if ok:
        print("[OK] All reference data files present.")
    return ok


# ===================================================================
# Spike data helpers
# ===================================================================

# Cell type constants (mirrors src/core/experiment_config.h)
_CELL_TYPES = {
    0: "Generic", 1: "KC", 2: "MBON", 3: "MBON", 4: "MBON",
    5: "DAN", 6: "DAN", 7: "PN", 8: "PN", 9: "LN", 10: "ORN",
    11: "FastSpiking", 12: "Bursting",
}

_REGION_MB_TYPES = {1, 2, 3, 4, 5, 6}   # KC, MBON*, DAN*
_REGION_AL_TYPES = {7, 8, 9, 10}         # PN*, LN, ORN


def _read_spikes(path: str):
    """Read spikes.bin and return (n_neurons, n_steps, times, spikes).

    ``spikes`` is a list of lists (step x neuron) of 0/1 values.
    Uses only the stdlib ``struct`` module to avoid a hard numpy
    dependency at import time; numpy is loaded lazily if available.
    """
    try:
        import numpy as np
        with open(path, "rb") as f:
            n_neurons, n_steps = struct.unpack("<II", f.read(8))
            times = []
            spike_list = []
            for _ in range(n_steps):
                t = struct.unpack("<f", f.read(4))[0]
                times.append(t)
                frame = np.frombuffer(f.read(n_neurons), dtype=np.uint8)
                spike_list.append(frame.copy())
        return n_neurons, n_steps, np.array(times), np.array(spike_list)
    except ImportError:
        # Pure-python fallback (slow but functional)
        with open(path, "rb") as f:
            n_neurons, n_steps = struct.unpack("<II", f.read(8))
            times = []
            spike_list = []
            for _ in range(n_steps):
                t = struct.unpack("<f", f.read(4))[0]
                times.append(t)
                raw = f.read(n_neurons)
                spike_list.append(list(raw))
        return n_neurons, n_steps, times, spike_list


def _read_neuron_types(path: str):
    """Read neurons.bin and return an array of cell-type integers.

    Format: [count:u32] then per neuron [root_id:u64, x:f32, y:f32,
    z:f32, type:u8].
    """
    types = []
    with open(path, "rb") as f:
        count = struct.unpack("<I", f.read(4))[0]
        record_size = 8 + 4 * 3 + 1  # u64 + 3*f32 + u8
        for _ in range(count):
            data = f.read(record_size)
            _rid, _x, _y, _z, ti = struct.unpack("<Q3fB", data)
            types.append(ti)
    return types


def _build_spike_summary(result_dir: str):
    """Build per-cell-type spike summaries from binary spike data.

    Returns a tuple of dicts suitable for passing to the validators:
    ``(firing_rate_sim, sparseness_sim, isi_sim, correlation_sim,
    sim_ms)``.

    If the binary files are missing, returns all-None and 0.0.
    """
    spike_path = os.path.join(result_dir, "spikes.bin")
    if not os.path.exists(spike_path):
        return None, None, None, None, 0.0

    n_neurons, n_steps, times, spikes = _read_spikes(spike_path)
    if n_steps < 2:
        return None, None, None, None, 0.0

    try:
        import numpy as np
        times_arr = np.asarray(times)
        spikes_arr = np.asarray(spikes)
    except ImportError:
        # Without numpy we cannot do detailed analysis
        return None, None, None, None, 0.0

    sim_ms = float(times_arr[-1] - times_arr[0])
    if sim_ms <= 0:
        return None, None, None, None, 0.0

    # Load neuron types if available
    neuron_types = [0] * n_neurons
    for candidate in [
        os.path.join(result_dir, "neurons.bin"),
        os.path.join(os.path.dirname(result_dir.rstrip("/")), "data", "neurons.bin"),
        os.path.join("data", "neurons.bin"),
    ]:
        if os.path.exists(candidate):
            loaded = _read_neuron_types(candidate)
            if len(loaded) >= n_neurons:
                neuron_types = loaded[:n_neurons]
            else:
                neuron_types = loaded + [0] * (n_neurons - len(loaded))
            break

    neuron_types_arr = np.array(neuron_types, dtype=np.uint8)

    # --- Firing rate summary ---
    per_neuron_counts = spikes_arr.sum(axis=0)
    fr_sim: dict = {}
    for type_id in np.unique(neuron_types_arr):
        label = _CELL_TYPES.get(int(type_id), f"type_{type_id}")
        if label in ("Generic", "FastSpiking", "Bursting", "ORN"):
            continue  # skip types without reference data
        mask = neuron_types_arr == type_id
        n = int(mask.sum())
        count = int(per_neuron_counts[mask].sum())
        if label in fr_sim:
            fr_sim[label]["n_neurons"] += n
            fr_sim[label]["spike_count"] += count
        else:
            fr_sim[label] = {"n_neurons": n, "spike_count": count}

    # --- Sparseness summary ---
    sp_sim: dict = {}
    mb_mask = np.isin(neuron_types_arr, list(_REGION_MB_TYPES))
    al_mask = np.isin(neuron_types_arr, list(_REGION_AL_TYPES))

    n_mb = int(mb_mask.sum())
    if n_mb > 0:
        mb_active_per_step = spikes_arr[:, mb_mask].any(axis=0)
        sp_sim["MB"] = {
            "n_active": int(mb_active_per_step.sum()),
            "n_total": n_mb,
        }

    n_al = int(al_mask.sum())
    if n_al > 0:
        al_active_per_step = spikes_arr[:, al_mask].any(axis=0)
        sp_sim["AL"] = {
            "n_active": int(al_active_per_step.sum()),
            "n_total": n_al,
        }

    # --- ISI summary ---
    cv_values = []
    for i in range(n_neurons):
        spike_times = times_arr[spikes_arr[:, i] > 0]
        if len(spike_times) < 3:
            continue
        isis = np.diff(spike_times)
        mu = isis.mean()
        if mu > 0:
            cv_values.append(float(isis.std() / mu))
    isi_sim = {"cv_values": cv_values}

    # --- Correlation summary ---
    corr_sim: dict = {"within": float("nan"), "between": float("nan")}
    max_sample = 200

    def _sample_corr(mat):
        n = mat.shape[1]
        if n < 2:
            return float("nan")
        if n > max_sample:
            idx = np.random.choice(n, max_sample, replace=False)
            mat = mat[:, idx]
        counts = mat.astype(np.float64)
        stds = counts.std(axis=0)
        valid = stds > 0
        if valid.sum() < 2:
            return 0.0
        cc = np.corrcoef(counts[:, valid].T)
        triu = np.triu_indices(cc.shape[0], k=1)
        return float(np.mean(cc[triu]))

    np.random.seed(42)
    within_vals = []
    if n_mb >= 2:
        within_vals.append(_sample_corr(spikes_arr[:, mb_mask]))
    if n_al >= 2:
        within_vals.append(_sample_corr(spikes_arr[:, al_mask]))
    within_finite = [v for v in within_vals if math.isfinite(v)]
    if within_finite:
        corr_sim["within"] = sum(within_finite) / len(within_finite)

    if n_mb >= 2 and n_al >= 2:
        n_s = min(max_sample // 2, n_mb, n_al)
        mb_idx = np.random.choice(n_mb, n_s, replace=False)
        al_idx = np.random.choice(n_al, n_s, replace=False)
        combined = np.hstack([
            spikes_arr[:, mb_mask][:, mb_idx],
            spikes_arr[:, al_mask][:, al_idx],
        ]).astype(np.float64)
        if combined.shape[0] > 10:
            cc = np.corrcoef(combined.T)
            cross = cc[:n_s, n_s:]
            corr_sim["between"] = float(np.nanmean(cross))

    return fr_sim, sp_sim, isi_sim, corr_sim, sim_ms


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FWMC validation dashboard: run all validators "
                    "and generate summary plots."
    )
    parser.add_argument(
        "result_dir",
        nargs="?",
        default=".",
        help="Path to results directory (default: current directory).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory for output PNGs (default: <result_dir>/plots).",
    )
    args = parser.parse_args()

    result_dir = os.path.abspath(args.result_dir)
    if not os.path.isdir(result_dir):
        print(f"Error: {result_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    # Step 1: source integrity
    print("=" * 60)
    print("  FWMC Validation Dashboard")
    print("=" * 60)
    print()
    check_sources()
    print()

    # Step 2: load simulation output
    metrics_path = os.path.join(result_dir, "metrics.csv")
    metrics = {}
    if os.path.exists(metrics_path):
        metrics = load_csv(metrics_path)
        n_rows = len(next(iter(metrics.values()))) if metrics else 0
        print(f"Loaded metrics.csv: {n_rows} rows, "
              f"{len(metrics)} columns.")
    else:
        print("[WARN] No metrics.csv found; will attempt binary files.",
              file=sys.stderr)

    # Step 3: detect applicable modules
    modules = detect_modules(metrics)
    print(f"Detected modules: {modules}")
    print()

    # Step 4: build spike summary from binary data
    fr_sim, sp_sim, isi_sim, corr_sim, sim_ms = _build_spike_summary(
        result_dir
    )

    # Fallback: try to get sim_ms from metrics.csv
    if sim_ms <= 0 and "time_ms" in metrics:
        t_vals = [v for v in metrics["time_ms"]
                  if isinstance(v, (int, float)) and math.isfinite(v)]
        if len(t_vals) >= 2:
            sim_ms = t_vals[-1] - t_vals[0]

    # Step 5: run validators
    fr_result = None
    sp_result = None
    isi_result = None
    corr_result = None

    if fr_sim is not None and sim_ms > 0:
        fr_result = validate_firing_rates(fr_sim, sim_ms)
        if fr_result:
            print(f"Firing rates: {len(fr_result)} cell types evaluated.")
    else:
        print("[SKIP] Firing rates: insufficient data.")

    if sp_sim is not None and sim_ms > 0:
        sp_result = validate_sparseness(sp_sim, sim_ms)
        if sp_result:
            print(f"Sparseness: {len(sp_result)} regions evaluated.")
    else:
        print("[SKIP] Sparseness: insufficient data.")

    if isi_sim is not None:
        isi_result = validate_isi(isi_sim, sim_ms)
        if isi_result:
            n_cv = len(isi_sim.get("cv_values", []))
            print(f"ISI: {n_cv} neurons with CV values.")
    else:
        print("[SKIP] ISI: insufficient data.")

    if corr_sim is not None:
        corr_result = validate_correlation(corr_sim, sim_ms)
        print("Correlation: evaluated.")
    else:
        print("[SKIP] Correlation: insufficient data.")

    print()

    # Step 6: print summary
    print_summary(
        firing_rates=fr_result,
        sparseness=sp_result,
        isi=isi_result,
        correlation=corr_result,
    )

    # Count pass/fail across all results
    all_pass = True
    total = 0
    n_pass = 0

    def _count(d):
        nonlocal all_pass, total, n_pass
        if d is None:
            return
        if isinstance(d, dict):
            for v in d.values():
                if isinstance(v, dict) and "pass" in v:
                    total += 1
                    if v["pass"]:
                        n_pass += 1
                    else:
                        all_pass = False
            if "pass" in d:
                total += 1
                if d["pass"]:
                    n_pass += 1
                else:
                    all_pass = False

    _count(fr_result)
    _count(sp_result)
    _count(isi_result)
    _count(corr_result)

    if total > 0:
        print(f"Overall: {n_pass}/{total} checks passed.")
    else:
        print("No validators could run (no spike data found).")

    # Step 7: generate plots
    if args.no_plots:
        print("Plot generation skipped (--no-plots).")
        sys.exit(0 if all_pass else 1)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.join(result_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    # Per-module PNGs
    modules_plotted = []

    if fr_result:
        fig_fr, ax_fr = plt.subplots(figsize=(7, 5))
        plot_firing_rate_panel(fr_result, ax_fr)
        fig_fr.tight_layout()
        path = os.path.join(out_dir, "firing_rates.png")
        fig_fr.savefig(path, dpi=150)
        plt.close(fig_fr)
        modules_plotted.append(("firing_rates", path))
        print(f"Saved: {path}")

    if sp_result:
        fig_sp, ax_sp = plt.subplots(figsize=(7, 5))
        plot_sparseness_panel(sp_result, ax_sp)
        fig_sp.tight_layout()
        path = os.path.join(out_dir, "sparseness.png")
        fig_sp.savefig(path, dpi=150)
        plt.close(fig_sp)
        modules_plotted.append(("sparseness", path))
        print(f"Saved: {path}")

    if isi_result:
        # Attach raw CV values for the histogram if available
        isi_plot_data = dict(isi_result)
        if isi_sim and "cv_values" in isi_sim:
            isi_plot_data["_cv_values"] = isi_sim["cv_values"]
        fig_isi, ax_isi = plt.subplots(figsize=(7, 5))
        plot_isi_panel(isi_plot_data, ax_isi)
        fig_isi.tight_layout()
        path = os.path.join(out_dir, "isi.png")
        fig_isi.savefig(path, dpi=150)
        plt.close(fig_isi)
        modules_plotted.append(("isi", path))
        print(f"Saved: {path}")

    if corr_result:
        fig_co, ax_co = plt.subplots(figsize=(7, 5))
        plot_correlation_panel(corr_result, ax_co)
        fig_co.tight_layout()
        path = os.path.join(out_dir, "correlation.png")
        fig_co.savefig(path, dpi=150)
        plt.close(fig_co)
        modules_plotted.append(("correlation", path))
        print(f"Saved: {path}")

    # Combined dashboard
    n_panels = max(len(modules_plotted), 1)
    ncols = 2
    nrows = (n_panels + ncols - 1) // ncols
    fig_dash, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    elif ncols == 1:
        axes = [[ax] for ax in axes]

    panel_funcs = {
        "firing_rates": (plot_firing_rate_panel, fr_result),
        "sparseness": (plot_sparseness_panel, sp_result),
        "isi": (plot_isi_panel, isi_plot_data if isi_result else None),
        "correlation": (plot_correlation_panel, corr_result),
    }

    idx = 0
    for name, _path in modules_plotted:
        r = idx // ncols
        c = idx % ncols
        func, data = panel_funcs.get(name, (None, None))
        if func and data:
            func(data, axes[r][c])
        idx += 1

    # Hide unused axes
    while idx < nrows * ncols:
        r = idx // ncols
        c = idx % ncols
        axes[r][c].set_visible(False)
        idx += 1

    fig_dash.suptitle("FWMC Validation Dashboard", fontsize=14,
                      fontweight="bold")
    fig_dash.tight_layout(rect=[0, 0, 1, 0.96])
    dash_path = os.path.join(out_dir, "validation_dashboard.png")
    fig_dash.savefig(dash_path, dpi=150)
    plt.close(fig_dash)
    print(f"Saved: {dash_path}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
