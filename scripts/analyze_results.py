#!/usr/bin/env python3
"""Analyze FWMC experiment results.

Reads the binary and CSV output files from a simulation run
and generates summary statistics and plots (if matplotlib available).

Usage:
    python3 scripts/analyze_results.py results/phase1_baseline
    python3 scripts/analyze_results.py results/phase2_shadow --plot
"""

import argparse
import csv
import os
import struct
import sys

import numpy as np


def read_spikes(path):
    """Read spikes.bin -> (n_neurons, n_steps, times, spike_matrix)"""
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
    """Read voltages.bin -> (n_neurons, n_steps, times, voltage_matrix)"""
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


def read_metrics(path):
    """Read metrics.csv -> dict of column arrays"""
    cols = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return cols
    for key in rows[0]:
        vals = []
        for row in rows:
            v = row[key]
            try:
                vals.append(float(v) if v else float("nan"))
            except ValueError:
                vals.append(float("nan"))
        cols[key] = np.array(vals)
    return cols


def read_per_neuron_error(path):
    """Read per_neuron_error.bin -> (n_neurons, n_steps, times, error_matrix)"""
    with open(path, "rb") as f:
        n_neurons, n_steps = struct.unpack("<II", f.read(8))
        times = []
        errors = []
        for _ in range(n_steps):
            t = struct.unpack("<f", f.read(4))[0]
            times.append(t)
            frame = np.frombuffer(f.read(n_neurons * 4), dtype=np.float32)
            errors.append(frame.copy())
    return n_neurons, n_steps, np.array(times), np.array(errors)


def print_summary(result_dir):
    """Print summary statistics for a result directory."""
    print(f"\n=== FWMC Results: {result_dir} ===\n")

    # Spikes
    spike_path = os.path.join(result_dir, "spikes.bin")
    if os.path.exists(spike_path):
        n_neurons, n_steps, times, spikes = read_spikes(spike_path)
        print(f"Spikes: {n_neurons} neurons, {n_steps} recorded steps")
        print(f"  Time range: {times[0]:.1f} - {times[-1]:.1f} ms")

        spike_counts = spikes.sum(axis=1)
        total_spikes = spikes.sum()
        print(f"  Total spikes: {total_spikes}")
        print(f"  Mean spike rate: {spike_counts.mean():.2f} spikes/step")
        print(f"  Peak spike rate: {spike_counts.max()} spikes/step")

        # Per-neuron firing rate
        per_neuron = spikes.sum(axis=0)
        duration_s = (times[-1] - times[0]) / 1000.0
        if duration_s > 0:
            rates_hz = per_neuron / duration_s
            active = (per_neuron > 0).sum()
            print(f"  Active neurons: {active}/{n_neurons} ({100*active/n_neurons:.1f}%)")
            print(f"  Mean firing rate: {rates_hz[rates_hz > 0].mean():.1f} Hz (active only)")
            print(f"  Max firing rate: {rates_hz.max():.1f} Hz")
    else:
        print("No spikes.bin found")

    # Voltages
    voltage_path = os.path.join(result_dir, "voltages.bin")
    if os.path.exists(voltage_path):
        n_neurons, n_steps, times, voltages = read_voltages(voltage_path)
        print(f"\nVoltages: {n_steps} recorded steps")
        print(f"  Mean voltage: {voltages.mean():.2f} mV")
        print(f"  Voltage range: [{voltages.min():.1f}, {voltages.max():.1f}] mV")

    # Metrics
    metrics_path = os.path.join(result_dir, "metrics.csv")
    if os.path.exists(metrics_path):
        metrics = read_metrics(metrics_path)
        print(f"\nMetrics: {len(next(iter(metrics.values())))} entries")
        for key, vals in metrics.items():
            valid = vals[~np.isnan(vals)]
            if len(valid) > 0:
                print(f"  {key}: mean={valid.mean():.4f}, min={valid.min():.4f}, max={valid.max():.4f}")

    # Per-neuron error
    error_path = os.path.join(result_dir, "per_neuron_error.bin")
    if os.path.exists(error_path):
        n_neurons, n_steps, times, errors = read_per_neuron_error(error_path)
        print(f"\nPer-neuron error: {n_steps} steps")
        print(f"  Mean error: {errors.mean():.4f}")
        print(f"  Max error: {errors.max():.4f}")
        worst = errors.mean(axis=0).argmax()
        print(f"  Worst neuron: #{worst} (mean error {errors.mean(axis=0)[worst]:.4f})")


def plot_results(result_dir):
    """Generate plots if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nInstall matplotlib for plots: pip install matplotlib")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"FWMC Results: {os.path.basename(result_dir)}")

    # 1. Raster plot
    spike_path = os.path.join(result_dir, "spikes.bin")
    if os.path.exists(spike_path):
        _, _, times, spikes = read_spikes(spike_path)
        ax = axes[0, 0]
        for step_idx in range(min(len(times), 1000)):
            t = times[step_idx]
            firing = np.where(spikes[step_idx])[0]
            if len(firing) > 0:
                ax.scatter([t] * len(firing), firing, s=0.5, c="black")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Neuron index")
        ax.set_title("Spike Raster")

    # 2. Spike count over time
    if os.path.exists(spike_path):
        ax = axes[0, 1]
        counts = spikes.sum(axis=1)
        ax.plot(times, counts, linewidth=0.5)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Spike count")
        ax.set_title("Population Activity")

    # 3. Metrics
    metrics_path = os.path.join(result_dir, "metrics.csv")
    if os.path.exists(metrics_path):
        metrics = read_metrics(metrics_path)
        ax = axes[1, 0]
        if "correlation" in metrics:
            valid_mask = ~np.isnan(metrics["correlation"])
            if valid_mask.any():
                ax.plot(metrics["time_ms"][valid_mask],
                        metrics["correlation"][valid_mask], label="Correlation")
        if "rmse" in metrics:
            valid_mask = ~np.isnan(metrics["rmse"])
            if valid_mask.any():
                ax.plot(metrics["time_ms"][valid_mask],
                        metrics["rmse"][valid_mask], label="RMSE")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Value")
        ax.set_title("Shadow Metrics")
        ax.legend()

    # 4. Voltage trace (first 5 neurons)
    voltage_path = os.path.join(result_dir, "voltages.bin")
    if os.path.exists(voltage_path):
        _, _, times_v, voltages = read_voltages(voltage_path)
        ax = axes[1, 1]
        for i in range(min(5, voltages.shape[1])):
            ax.plot(times_v, voltages[:, i], linewidth=0.5, label=f"N{i}")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Voltage (mV)")
        ax.set_title("Voltage Traces")
        ax.legend(fontsize=8)
    else:
        axes[1, 1].text(0.5, 0.5, "No voltage data", ha="center", va="center",
                        transform=axes[1, 1].transAxes)

    plt.tight_layout()
    out_path = os.path.join(result_dir, "analysis.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to: {out_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analyze FWMC experiment results")
    parser.add_argument("result_dir", help="Path to results directory")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    args = parser.parse_args()

    if not os.path.isdir(args.result_dir):
        print(f"Error: {args.result_dir} is not a directory")
        sys.exit(1)

    print_summary(args.result_dir)

    if args.plot:
        plot_results(args.result_dir)


if __name__ == "__main__":
    main()
