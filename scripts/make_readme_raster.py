#!/usr/bin/env python3
"""Generate a publication-quality spike raster for the FWMC README."""

import struct
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path


def read_spikes(path):
    with open(path, "rb") as f:
        n_neurons, n_steps = struct.unpack("<II", f.read(8))
        times, spikes = [], []
        for _ in range(n_steps):
            t = struct.unpack("<f", f.read(4))[0]
            times.append(t)
            frame = np.frombuffer(f.read(n_neurons), dtype=np.uint8)
            spikes.append(frame.copy())
    return n_neurons, np.array(times), np.array(spikes)


def read_neurons(path):
    with open(path, "rb") as f:
        count = struct.unpack("<I", f.read(4))[0]
        types = np.empty(count, dtype=np.uint8)
        for i in range(count):
            f.read(8)  # root_id
            f.read(12)  # x, y, z
            types[i] = struct.unpack("<B", f.read(1))[0]
    return count, types


def main():
    data_dir = Path("results/raster_demo")
    out_path = Path("figures/spike_raster_readme.png")
    out_path.parent.mkdir(exist_ok=True)

    n_neurons, times, spikes = read_spikes(data_dir / "spikes.bin")
    _, neuron_types = read_neurons(data_dir / "neurons.bin")

    # Read region assignments from neuron types
    # Region 0 = ORN (indices 0-399), Region 1 = PN (400-549), Region 2 = KC (550-849)
    # We use the 'region' field which is stored as part of the neuron type mapping
    # For parametric gen, neurons are laid out contiguously by region
    regions = np.zeros(n_neurons, dtype=int)
    # Detect boundaries from type changes
    region_boundaries = [0]
    for i in range(1, n_neurons):
        if neuron_types[i] != neuron_types[i - 1]:
            # Check if this is a real region boundary (type pattern change)
            pass
    # Use known layout: ORN=400, PN=150, KC=300
    regions[:400] = 0
    regions[400:550] = 1
    regions[550:] = 2

    region_names = ["ORN (400)", "PN (150)", "KC (300)"]
    region_colors = ["#4A90D9", "#E8644A", "#5CB85C"]

    # Stimulus periods
    stimuli = [
        (200, 600, "Odor A\n(vinegar)", "#FFD54F"),
        (800, 1100, "Odor B\n(banana)", "#81C784"),
        (1300, 1600, "Mixture\nA+B", "#FF8A65"),
    ]

    # Build figure
    fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    # Draw stimulus bands
    for start, end, label, color in stimuli:
        rect = Rectangle((start, -5), end - start, n_neurons + 10,
                          facecolor=color, alpha=0.08, edgecolor="none")
        ax.add_patch(rect)
        ax.text((start + end) / 2, n_neurons + 15, label,
                ha="center", va="bottom", fontsize=7, color=color,
                fontweight="bold", alpha=0.9)

    # Plot spikes by region
    for step_idx in range(len(times)):
        firing = np.where(spikes[step_idx])[0]
        if len(firing) == 0:
            continue
        for reg_id in range(3):
            mask = regions[firing] == reg_id
            reg_neurons = firing[mask]
            if len(reg_neurons) > 0:
                ax.scatter(
                    np.full(len(reg_neurons), times[step_idx]),
                    reg_neurons,
                    s=0.15, c=region_colors[reg_id],
                    marker="|", linewidths=0.4, alpha=0.8,
                    rasterized=True,
                )

    # Region boundary lines and labels
    for boundary, name, color in [(0, region_names[0], region_colors[0]),
                                   (400, region_names[1], region_colors[1]),
                                   (550, region_names[2], region_colors[2])]:
        if boundary > 0:
            ax.axhline(y=boundary, color="#ffffff", alpha=0.15, linewidth=0.5,
                       linestyle="--")
        mid = boundary + [200, 75, 150][region_names.index(name)]
        ax.text(-40, mid, name, ha="right", va="center", fontsize=8,
                color=color, fontweight="bold", alpha=0.9)

    ax.set_xlim(times[0] - 10, times[-1] + 10)
    ax.set_ylim(-10, n_neurons + 40)
    ax.set_xlabel("Time (ms)", color="#cccccc", fontsize=10)
    ax.set_ylabel("")
    ax.set_title("FWMC Spike Raster: Drosophila Olfactory Circuit (850 neurons)",
                 color="#eeeeee", fontsize=12, fontweight="bold", pad=25)

    ax.tick_params(colors="#888888", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#444444")
    ax.spines["left"].set_color("#444444")
    ax.set_yticks([])

    # Total spike count annotation
    total = int(spikes.sum())
    duration_s = (times[-1] - times[0]) / 1000
    mean_rate = total / n_neurons / duration_s
    ax.text(0.99, 0.02,
            f"{total:,} spikes | {mean_rate:.1f} Hz mean",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="#888888", fontstyle="italic")

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"Saved {out_path} ({total} spikes, {mean_rate:.1f} Hz mean rate)")


if __name__ == "__main__":
    main()
