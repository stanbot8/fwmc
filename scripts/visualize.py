#!/usr/bin/env python3
"""Comprehensive visualization for FWMC spiking neural network simulations.

Generates publication-quality plots from binary simulation output.

Usage:
    python3 scripts/visualize.py results/ --output figures/
    python3 scripts/visualize.py results/ --output figures/ --data data/
    python3 scripts/visualize.py results/ --output figures/ --checkpoint results/checkpoint.bin

Plots generated (each saved as separate high-DPI PNG + combined summary):
  1. spike_raster.png         : color-coded by neuron region
  2. firing_rate_heatmap.png  : time x neuron heatmap (50ms sliding window)
  3. population_rates.png     : per-region stacked area chart
  4. connectivity_matrix.png  : adjacency matrix (subsampled if >1000)
  5. weight_distribution.png  : synaptic weight histogram (before/after)
  6. phase_plot.png           : Izhikevich v vs u scatter
  7. isi_distribution.png     : inter-spike interval histogram (log scale)
  8. correlation_matrix.png   : neuron-neuron spike correlation
  9. brain_3d.png             : 3D scatter of neuron positions by firing rate
"""

import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Binary readers (matching recorder.h and connectome_loader.h)
# ---------------------------------------------------------------------------

def read_spikes(path):
    """Read spikes.bin -> (n_neurons, n_steps, times, spike_matrix[steps, neurons])"""
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


def read_voltages(path):
    """Read voltages.bin -> (n_neurons, n_steps, times, voltage_matrix[steps, neurons])"""
    with open(path, "rb") as f:
        n_neurons, n_steps = struct.unpack("<II", f.read(8))
        times = []
        voltages = []
        for _ in range(n_steps):
            t = struct.unpack("<f", f.read(4))[0]
            times.append(t)
            frame = np.frombuffer(f.read(n_neurons * 4), dtype=np.float32)
            voltages.append(frame.copy())
    return n_neurons, n_steps, np.array(times), np.array(voltages)


def read_neurons(path):
    """Read neurons.bin -> dict with root_id, x, y, z, type arrays."""
    with open(path, "rb") as f:
        count = struct.unpack("<I", f.read(4))[0]
        root_ids = np.empty(count, dtype=np.uint64)
        x = np.empty(count, dtype=np.float32)
        y = np.empty(count, dtype=np.float32)
        z = np.empty(count, dtype=np.float32)
        types = np.empty(count, dtype=np.uint8)
        for i in range(count):
            root_ids[i] = struct.unpack("<Q", f.read(8))[0]
            x[i], y[i], z[i] = struct.unpack("<fff", f.read(12))
            types[i] = struct.unpack("<B", f.read(1))[0]
    return {
        "count": count,
        "root_id": root_ids,
        "x": x, "y": y, "z": z,
        "type": types,
    }


def read_synapses(path):
    """Read synapses.bin -> dict with pre, post, weight, nt arrays."""
    with open(path, "rb") as f:
        count = struct.unpack("<I", f.read(4))[0]
        pre = np.empty(count, dtype=np.uint32)
        post = np.empty(count, dtype=np.uint32)
        weight = np.empty(count, dtype=np.float32)
        nt = np.empty(count, dtype=np.uint8)
        for i in range(count):
            pre[i], post[i] = struct.unpack("<II", f.read(8))
            weight[i] = struct.unpack("<f", f.read(4))[0]
            nt[i] = struct.unpack("<B", f.read(1))[0]
    return {
        "count": count,
        "pre": pre, "post": post,
        "weight": weight, "nt": nt,
    }


def read_synapses_fast(path):
    """Read synapses.bin using bulk reads for large files."""
    with open(path, "rb") as f:
        count = struct.unpack("<I", f.read(4))[0]
        # Each record: u32 pre, u32 post, f32 weight, u8 nt = 13 bytes
        record_size = 13
        raw = f.read(count * record_size)

    if len(raw) < count * record_size:
        # Fall back to per-record reader
        return read_synapses(path)

    pre = np.empty(count, dtype=np.uint32)
    post = np.empty(count, dtype=np.uint32)
    weight = np.empty(count, dtype=np.float32)
    nt = np.empty(count, dtype=np.uint8)

    for i in range(count):
        off = i * record_size
        pre[i], post[i] = struct.unpack_from("<II", raw, off)
        weight[i] = struct.unpack_from("<f", raw, off + 8)[0]
        nt[i] = raw[off + 12]

    return {
        "count": count,
        "pre": pre, "post": post,
        "weight": weight, "nt": nt,
    }


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def compute_firing_rates(spikes, times, window_ms=50.0):
    """Compute firing rate per neuron in a sliding window.

    Returns: rate_matrix[steps, neurons] in Hz.
    """
    n_steps, n_neurons = spikes.shape
    if n_steps < 2:
        return np.zeros_like(spikes, dtype=np.float64)

    dt = float(np.median(np.diff(times)))
    if dt <= 0:
        dt = 0.1
    win_steps = max(1, int(window_ms / dt))
    cumsum = np.cumsum(spikes.astype(np.float64), axis=0)
    padded = np.vstack([np.zeros((1, n_neurons)), cumsum])
    # Sliding window sum
    rates = np.zeros_like(cumsum)
    for i in range(n_steps):
        lo = max(0, i + 1 - win_steps)
        rates[i] = (padded[i + 1] - padded[lo]) / (window_ms / 1000.0)
    return rates


def get_region_labels(neuron_types, n_neurons):
    """Build region array from neuron type field (used as region proxy)."""
    if neuron_types is not None and len(neuron_types) == n_neurons:
        return neuron_types
    return np.zeros(n_neurons, dtype=np.uint8)


def safe_subsample(n, max_n):
    """Return sorted indices for subsampling if n > max_n."""
    if n <= max_n:
        return np.arange(n)
    return np.sort(np.random.default_rng(42).choice(n, max_n, replace=False))


# ---------------------------------------------------------------------------
# Plot functions: each is self-contained and safe to call independently
# ---------------------------------------------------------------------------

def setup_style():
    """Configure matplotlib for publication quality."""
    import matplotlib.pyplot as plt
    available = plt.style.available
    for s in ["seaborn-v0_8-whitegrid", "seaborn-whitegrid"]:
        if s in available:
            plt.style.use(s)
            return
    plt.rcParams.update({
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 10,
    })


def plot_spike_raster(spikes, times, regions, output_dir):
    """1. Spike raster plot color-coded by neuron region."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    fig, ax = plt.subplots(figsize=(14, 6))

    unique_regions = np.unique(regions)
    n_regions = len(unique_regions)
    cmap = plt.cm.viridis
    norm = Normalize(vmin=unique_regions.min(), vmax=max(unique_regions.max(), 1))

    # Collect spike events
    for step_idx in range(len(times)):
        firing = np.where(spikes[step_idx])[0]
        if len(firing) == 0:
            continue
        colors = cmap(norm(regions[firing]))
        ax.scatter(
            np.full(len(firing), times[step_idx]),
            firing,
            s=0.3, c=colors, marker="|", linewidths=0.5, rasterized=True
        )

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron index")
    ax.set_title("Spike Raster (color = region)")
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label="Region", pad=0.01)
    if n_regions <= 10:
        cbar.set_ticks(unique_regions)

    fig.tight_layout()
    path = os.path.join(output_dir, "spike_raster.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return fig


def plot_firing_rate_heatmap(spikes, times, regions, output_dir, window_ms=50.0):
    """2. Firing rate heatmap: time x neuron, neurons sorted by region."""
    import matplotlib.pyplot as plt

    rates = compute_firing_rates(spikes, times, window_ms)
    sort_idx = np.argsort(regions)
    rates_sorted = rates[:, sort_idx]

    # Subsample time if very long
    max_time_bins = 2000
    if rates_sorted.shape[0] > max_time_bins:
        step = rates_sorted.shape[0] // max_time_bins
        rates_sorted = rates_sorted[::step]
        plot_times = times[::step]
    else:
        plot_times = times

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(
        rates_sorted.T,
        aspect="auto",
        origin="lower",
        extent=[plot_times[0], plot_times[-1], 0, rates_sorted.shape[1]],
        cmap="hot",
        interpolation="nearest",
    )
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron (sorted by region)")
    ax.set_title(f"Firing Rate Heatmap ({window_ms:.0f}ms window)")
    fig.colorbar(im, ax=ax, label="Firing rate (Hz)", pad=0.01)

    fig.tight_layout()
    path = os.path.join(output_dir, "firing_rate_heatmap.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return fig


def plot_population_rates(spikes, times, regions, output_dir, window_ms=50.0):
    """3. Per-region population firing rates: stacked area chart."""
    import matplotlib.pyplot as plt

    unique_regions = np.unique(regions)
    rates = compute_firing_rates(spikes, times, window_ms)

    # Mean firing rate per region per timestep
    region_rates = {}
    for r in unique_regions:
        mask = regions == r
        region_rates[r] = rates[:, mask].mean(axis=1)

    fig, ax = plt.subplots(figsize=(14, 5))
    labels = [f"Region {r}" for r in unique_regions]
    stacks = np.array([region_rates[r] for r in unique_regions])
    ax.stackplot(times, stacks, labels=labels, alpha=0.8)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Mean firing rate (Hz)")
    ax.set_title("Population Firing Rates by Region")
    ax.legend(loc="upper right", fontsize=8, ncol=min(len(unique_regions), 4))

    fig.tight_layout()
    path = os.path.join(output_dir, "population_rates.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return fig


def plot_connectivity_matrix(synapse_data, n_neurons, output_dir, max_display=1000):
    """4. N x N adjacency matrix visualization."""
    import matplotlib.pyplot as plt

    pre = synapse_data["pre"]
    post = synapse_data["post"]
    weight = synapse_data["weight"]

    idx = safe_subsample(n_neurons, max_display)
    n_sub = len(idx)
    idx_set = set(idx.tolist())
    # Build mapping from original index to display index
    idx_map = {int(v): i for i, v in enumerate(idx)}

    mat = np.zeros((n_sub, n_sub), dtype=np.float32)
    for k in range(len(pre)):
        p, q = int(pre[k]), int(post[k])
        if p in idx_map and q in idx_map:
            mat[idx_map[p], idx_map[q]] += weight[k]

    fig, ax = plt.subplots(figsize=(8, 8))
    vmax = np.percentile(np.abs(mat[mat != 0]), 99) if np.any(mat != 0) else 1.0
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   aspect="auto", interpolation="nearest")
    ax.set_xlabel("Postsynaptic neuron")
    ax.set_ylabel("Presynaptic neuron")
    title = "Connectivity Matrix"
    if n_neurons > max_display:
        title += f" (subsampled {max_display}/{n_neurons})"
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Synaptic weight", pad=0.01)

    fig.tight_layout()
    path = os.path.join(output_dir, "connectivity_matrix.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return fig


def plot_weight_distribution(synapse_data, checkpoint_synapse_data, output_dir):
    """5. Histogram of synaptic weights before/after."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))

    weights_init = synapse_data["weight"]
    ax.hist(weights_init, bins=100, alpha=0.6, label="Initial", color="#2196F3",
            density=True, edgecolor="none")

    if checkpoint_synapse_data is not None:
        weights_final = checkpoint_synapse_data["weight"]
        ax.hist(weights_final, bins=100, alpha=0.6, label="Final (checkpoint)",
                color="#FF5722", density=True, edgecolor="none")

    ax.set_xlabel("Synaptic weight")
    ax.set_ylabel("Density")
    ax.set_title("Synaptic Weight Distribution")
    ax.legend()

    fig.tight_layout()
    path = os.path.join(output_dir, "weight_distribution.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return fig


def plot_phase(voltages, times, output_dir, timepoint_idx=None):
    """6. Phase plot: v vs u (approximated from dv/dt) at a single timepoint.

    Since u is not directly recorded, we approximate it from the Izhikevich
    equation:  u ~ 0.04*v^2 + 5*v + 140 - dv/dt
    If multiple voltage timesteps exist, we use finite differences.
    """
    import matplotlib.pyplot as plt

    n_steps, n_neurons = voltages.shape
    if n_steps < 2:
        print("  Skipping phase plot: not enough voltage timesteps")
        return None

    if timepoint_idx is None:
        timepoint_idx = n_steps // 2  # mid-simulation

    dt = float(times[1] - times[0]) if len(times) > 1 else 0.1
    if dt <= 0:
        dt = 0.1

    v = voltages[timepoint_idx]
    # Approximate dv/dt from surrounding steps
    if timepoint_idx > 0 and timepoint_idx < n_steps - 1:
        dvdt = (voltages[timepoint_idx + 1] - voltages[timepoint_idx - 1]) / (2 * dt)
    elif timepoint_idx > 0:
        dvdt = (voltages[timepoint_idx] - voltages[timepoint_idx - 1]) / dt
    else:
        dvdt = (voltages[timepoint_idx + 1] - voltages[timepoint_idx]) / dt

    # From Izhikevich: dv/dt = 0.04v^2 + 5v + 140 - u + I
    # Approximate: u ~ 0.04v^2 + 5v + 140 - dv/dt  (ignoring I)
    u_approx = 0.04 * v**2 + 5.0 * v + 140.0 - dvdt

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(v, u_approx, s=2, alpha=0.5, c=v, cmap="coolwarm", rasterized=True)
    ax.set_xlabel("Membrane potential v (mV)")
    ax.set_ylabel("Recovery variable u (approx)")
    ax.set_title(f"Izhikevich Phase Plane (t = {times[timepoint_idx]:.1f} ms)")

    # Draw nullclines
    v_range = np.linspace(v.min() - 5, min(v.max() + 5, 40), 300)
    v_null = 0.04 * v_range**2 + 5 * v_range + 140  # dv/dt=0 => u = 0.04v^2+5v+140
    ax.plot(v_range, v_null, "k--", alpha=0.4, linewidth=1, label="v-nullcline")
    ax.legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(output_dir, "phase_plot.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return fig


def plot_isi_distribution(spikes, times, output_dir):
    """7. Inter-spike interval histogram (log scale)."""
    import matplotlib.pyplot as plt

    n_steps, n_neurons = spikes.shape
    isis = []
    for neuron_idx in range(n_neurons):
        spike_times = times[spikes[:, neuron_idx] > 0]
        if len(spike_times) > 1:
            isis.extend(np.diff(spike_times).tolist())

    if not isis:
        print("  Skipping ISI distribution: no inter-spike intervals found")
        return None

    isis = np.array(isis)
    isis = isis[isis > 0]

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.logspace(np.log10(max(isis.min(), 0.01)), np.log10(isis.max()), 80)
    ax.hist(isis, bins=bins, color="#4CAF50", edgecolor="none", alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Inter-spike interval (ms)")
    ax.set_ylabel("Count")
    ax.set_title(f"ISI Distribution (n = {len(isis)} intervals)")

    # Annotate statistics
    median_isi = np.median(isis)
    ax.axvline(median_isi, color="red", linestyle="--", alpha=0.7,
               label=f"Median = {median_isi:.2f} ms")
    ax.legend()

    fig.tight_layout()
    path = os.path.join(output_dir, "isi_distribution.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return fig


def plot_correlation_matrix(spikes, times, output_dir, max_neurons=200, window_ms=50.0):
    """8. Neuron-neuron spike correlation matrix."""
    import matplotlib.pyplot as plt
    from scipy.ndimage import uniform_filter1d

    n_steps, n_neurons = spikes.shape
    idx = safe_subsample(n_neurons, max_neurons)
    sub_spikes = spikes[:, idx].astype(np.float64)

    # Smooth with a window to get rate-based correlation
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.1
    win = max(1, int(window_ms / dt))
    smoothed = uniform_filter1d(sub_spikes, size=win, axis=0)

    # Correlation matrix
    # Center the signals
    smoothed -= smoothed.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(smoothed, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    smoothed /= norms
    corr = smoothed.T @ smoothed

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1,
                   aspect="auto", interpolation="nearest")
    title = "Spike Correlation Matrix"
    if n_neurons > max_neurons:
        title += f" ({max_neurons}/{n_neurons} neurons)"
    ax.set_title(title)
    ax.set_xlabel("Neuron")
    ax.set_ylabel("Neuron")
    fig.colorbar(im, ax=ax, label="Pearson r", pad=0.01)

    fig.tight_layout()
    path = os.path.join(output_dir, "correlation_matrix.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return fig


def plot_brain_3d(neuron_data, spikes, times, output_dir):
    """9. 3D scatter of neuron positions colored by firing rate."""
    import matplotlib.pyplot as plt

    x, y, z = neuron_data["x"], neuron_data["y"], neuron_data["z"]
    # Check if positions are meaningful (not all zero)
    if np.all(x == 0) and np.all(y == 0) and np.all(z == 0):
        print("  Skipping 3D brain plot: all neuron positions are zero")
        return None

    n_neurons = neuron_data["count"]
    duration_s = (times[-1] - times[0]) / 1000.0 if len(times) > 1 else 1.0
    if duration_s <= 0:
        duration_s = 1.0

    # Per-neuron total spikes -> firing rate
    if spikes is not None and spikes.shape[1] == n_neurons:
        rates = spikes.sum(axis=0).astype(np.float64) / duration_s
    else:
        rates = np.zeros(n_neurons)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(x, y, z, c=rates, cmap="plasma", s=2, alpha=0.6,
                    rasterized=True)
    ax.set_xlabel("X (nm)")
    ax.set_ylabel("Y (nm)")
    ax.set_zlabel("Z (nm)")
    ax.set_title("3D Neuron Positions (color = firing rate Hz)")
    fig.colorbar(sc, ax=ax, label="Firing rate (Hz)", shrink=0.6, pad=0.1)

    fig.tight_layout()
    path = os.path.join(output_dir, "brain_3d.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return fig


# ---------------------------------------------------------------------------
# Combined summary figure
# ---------------------------------------------------------------------------

def make_summary(figs, output_dir):
    """Combine individual figures into a single multi-panel summary."""
    import matplotlib.pyplot as plt
    from matplotlib.image import imread

    # Collect the saved PNGs
    names = [
        "spike_raster", "firing_rate_heatmap", "population_rates",
        "connectivity_matrix", "weight_distribution", "phase_plot",
        "isi_distribution", "correlation_matrix", "brain_3d",
    ]
    images = []
    for name in names:
        path = os.path.join(output_dir, f"{name}.png")
        if os.path.exists(path):
            images.append((name, imread(path)))

    if not images:
        print("  No individual plots to combine into summary")
        return

    n = len(images)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 5))
    if rows == 1:
        axes = axes.reshape(1, -1) if n > 1 else np.array([[axes]])
    for i, (name, img) in enumerate(images):
        r, c = divmod(i, cols)
        axes[r, c].imshow(img)
        axes[r, c].set_title(name.replace("_", " ").title(), fontsize=10)
        axes[r, c].axis("off")
    # Hide unused axes
    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].axis("off")

    fig.suptitle("FWMC Simulation Summary", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(output_dir, "summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality plots from FWMC simulation output"
    )
    parser.add_argument("result_dir", help="Directory containing spikes.bin, voltages.bin, etc.")
    parser.add_argument("--output", "-o", default="figures",
                        help="Output directory for plots (default: figures)")
    parser.add_argument("--data", "-d", default="data",
                        help="Data directory with neurons.bin and synapses.bin (default: data)")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint synapses.bin for weight comparison")
    parser.add_argument("--dpi", type=int, default=200, help="Output DPI (default: 200)")
    parser.add_argument("--window", type=float, default=50.0,
                        help="Sliding window for firing rate in ms (default: 50)")
    args = parser.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib is required. Install with: pip install matplotlib")
        sys.exit(1)

    setup_style()

    os.makedirs(args.output, exist_ok=True)
    print(f"FWMC Visualization")
    print(f"  Results dir : {args.result_dir}")
    print(f"  Data dir    : {args.data}")
    print(f"  Output dir  : {args.output}")
    print()

    # ---- Load available data ----

    spikes = times = None
    n_neurons = 0
    spike_path = os.path.join(args.result_dir, "spikes.bin")
    if os.path.exists(spike_path):
        print("Loading spikes.bin...")
        n_neurons, n_steps, times, spikes = read_spikes(spike_path)
        print(f"  {n_neurons} neurons, {n_steps} steps, "
              f"t=[{times[0]:.1f}, {times[-1]:.1f}] ms")
    else:
        print("No spikes.bin found; many plots will be skipped")

    voltages = times_v = None
    voltage_path = os.path.join(args.result_dir, "voltages.bin")
    if os.path.exists(voltage_path):
        print("Loading voltages.bin...")
        _, _, times_v, voltages = read_voltages(voltage_path)
        print(f"  {voltages.shape[1]} neurons, {voltages.shape[0]} steps")

    neuron_data = None
    neuron_path = os.path.join(args.data, "neurons.bin")
    if os.path.exists(neuron_path):
        print("Loading neurons.bin...")
        neuron_data = read_neurons(neuron_path)
        print(f"  {neuron_data['count']} neurons")
    # Also try result_dir for neurons
    elif os.path.exists(os.path.join(args.result_dir, "neurons.bin")):
        neuron_path = os.path.join(args.result_dir, "neurons.bin")
        print(f"Loading neurons.bin from result dir...")
        neuron_data = read_neurons(neuron_path)
        print(f"  {neuron_data['count']} neurons")

    synapse_data = None
    synapse_path = os.path.join(args.data, "synapses.bin")
    if os.path.exists(synapse_path):
        print("Loading synapses.bin...")
        synapse_data = read_synapses_fast(synapse_path)
        print(f"  {synapse_data['count']} synapses")
    elif os.path.exists(os.path.join(args.result_dir, "synapses.bin")):
        synapse_path = os.path.join(args.result_dir, "synapses.bin")
        print(f"Loading synapses.bin from result dir...")
        synapse_data = read_synapses_fast(synapse_path)
        print(f"  {synapse_data['count']} synapses")

    checkpoint_synapse_data = None
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint synapses: {args.checkpoint}")
        checkpoint_synapse_data = read_synapses_fast(args.checkpoint)

    # Build region array
    if neuron_data is not None and n_neurons > 0:
        regions = neuron_data["type"][:n_neurons]
    else:
        regions = np.zeros(max(n_neurons, 1), dtype=np.uint8)

    print()
    print("Generating plots...")

    # ---- 1. Spike raster ----
    if spikes is not None:
        try:
            plot_spike_raster(spikes, times, regions, args.output)
        except Exception as e:
            print(f"  [WARN] Spike raster failed: {e}")

    # ---- 2. Firing rate heatmap ----
    if spikes is not None:
        try:
            plot_firing_rate_heatmap(spikes, times, regions, args.output,
                                     window_ms=args.window)
        except Exception as e:
            print(f"  [WARN] Firing rate heatmap failed: {e}")

    # ---- 3. Population rates ----
    if spikes is not None:
        try:
            plot_population_rates(spikes, times, regions, args.output,
                                   window_ms=args.window)
        except Exception as e:
            print(f"  [WARN] Population rates failed: {e}")

    # ---- 4. Connectivity matrix ----
    if synapse_data is not None:
        nn = neuron_data["count"] if neuron_data else n_neurons
        if nn > 0:
            try:
                plot_connectivity_matrix(synapse_data, nn, args.output)
            except Exception as e:
                print(f"  [WARN] Connectivity matrix failed: {e}")

    # ---- 5. Weight distribution ----
    if synapse_data is not None:
        try:
            plot_weight_distribution(synapse_data, checkpoint_synapse_data,
                                      args.output)
        except Exception as e:
            print(f"  [WARN] Weight distribution failed: {e}")

    # ---- 6. Phase plot ----
    if voltages is not None:
        try:
            plot_phase(voltages, times_v, args.output)
        except Exception as e:
            print(f"  [WARN] Phase plot failed: {e}")

    # ---- 7. ISI distribution ----
    if spikes is not None:
        try:
            plot_isi_distribution(spikes, times, args.output)
        except Exception as e:
            print(f"  [WARN] ISI distribution failed: {e}")

    # ---- 8. Correlation matrix ----
    if spikes is not None:
        try:
            plot_correlation_matrix(spikes, times, args.output,
                                     window_ms=args.window)
        except Exception as e:
            print(f"  [WARN] Correlation matrix failed: {e}")

    # ---- 9. 3D brain plot ----
    if neuron_data is not None:
        try:
            plot_brain_3d(neuron_data, spikes, times, args.output)
        except Exception as e:
            print(f"  [WARN] 3D brain plot failed: {e}")

    # ---- Combined summary ----
    print()
    print("Generating summary panel...")
    try:
        make_summary([], args.output)
    except Exception as e:
        print(f"  [WARN] Summary panel failed: {e}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
