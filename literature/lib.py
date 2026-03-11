#!/usr/bin/env python3
"""Shared validation utilities for FWMC literature comparisons.

Provides low-level helpers (CSV I/O, interpolation, RMSE), neuroscience
validators (firing rates, sparseness, ISI, correlation), plot functions,
and a summary printer.  All validators compare FWMC simulation output
against consensus reference data derived from published Drosophila
electrophysiology literature.

Adapted from skibidy's validation pattern.
"""

import csv
import math
import os
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Plot colours (matching skibidy)
# ---------------------------------------------------------------------------
SIM_COLOR = "#D46664"
REF_COLOR = "#4A90D9"

# ---------------------------------------------------------------------------
# Reference data directory
# ---------------------------------------------------------------------------
_DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"


def _ref_path(filename: str) -> pathlib.Path:
    """Resolve a consensus CSV inside literature/data/."""
    p = _DATA_DIR / filename
    if not p.exists():
        raise FileNotFoundError(f"Reference CSV not found: {p}")
    return p


# ===================================================================
# 1. Low-level helpers
# ===================================================================

def plots_dir(csv_path: str) -> str:
    """Derive a plots output directory next to the given CSV path.

    Example: ``results/run1/metrics.csv`` -> ``results/run1/plots``
    """
    parent = os.path.dirname(os.path.abspath(csv_path))
    out = os.path.join(parent, "plots")
    os.makedirs(out, exist_ok=True)
    return out


def load_csv(path: str) -> dict:
    """Load a CSV file into a dict of lists (column-oriented).

    Each key is a header name.  Values are lists of floats where
    possible, otherwise kept as strings.
    """
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    if not rows:
        return {}
    result: dict = {k: [] for k in rows[0]}
    for row in rows:
        for k, v in row.items():
            try:
                result[k].append(float(v))
            except (ValueError, TypeError):
                result[k].append(v)
    return result


def interpolate(x_ref: list, y_ref: list, x_query: float) -> float:
    """Piecewise linear interpolation.

    ``x_ref`` must be sorted ascending.  Values outside the range are
    clamped to the nearest endpoint.
    """
    if not x_ref:
        raise ValueError("x_ref is empty")
    if x_query <= x_ref[0]:
        return y_ref[0]
    if x_query >= x_ref[-1]:
        return y_ref[-1]
    for i in range(len(x_ref) - 1):
        if x_ref[i] <= x_query <= x_ref[i + 1]:
            t = (x_query - x_ref[i]) / (x_ref[i + 1] - x_ref[i])
            return y_ref[i] + t * (y_ref[i + 1] - y_ref[i])
    return y_ref[-1]


def peak_normalize(values: list) -> list:
    """Normalize a list of numbers by its maximum absolute value.

    Returns a new list.  If the peak is zero, returns all zeros.
    """
    peak = max(abs(v) for v in values) if values else 0.0
    if peak == 0.0:
        return [0.0] * len(values)
    return [v / peak for v in values]


def compute_rmse(sim: list, ref: list) -> float:
    """Root-mean-square error between two equal-length sequences."""
    if len(sim) != len(ref):
        raise ValueError(
            f"Length mismatch: sim={len(sim)}, ref={len(ref)}"
        )
    if not sim:
        return 0.0
    mse = sum((s - r) ** 2 for s, r in zip(sim, ref)) / len(sim)
    return math.sqrt(mse)


# ===================================================================
# 2. Reference data loading
# ===================================================================

def _load_firing_rate_reference() -> dict:
    """Load firing_rates_by_cell_type.csv into a lookup dict.

    Returns ``{cell_type: [(min_hz, max_hz, condition, source), ...]}``
    grouped by cell type.
    """
    path = _ref_path("firing_rates_by_cell_type.csv")
    raw = load_csv(str(path))
    result: dict = {}
    n = len(raw.get("cell_type", []))
    for i in range(n):
        ct = raw["cell_type"][i]
        entry = (
            float(raw["min_hz"][i]),
            float(raw["max_hz"][i]),
            raw["condition"][i],
            raw["source"][i],
        )
        result.setdefault(ct, []).append(entry)
    return result


# ===================================================================
# 3. Neuroscience validators
# ===================================================================

def validate_firing_rates(sim: dict, sim_ms: float) -> dict:
    """Compare per-cell-type firing rates against literature ranges.

    Parameters
    ----------
    sim : dict
        Pre-computed summary with per-cell-type spike counts.
        Expected keys per cell type: ``{<cell_type>: {"spike_count": int,
        "n_neurons": int}}``.  Alternatively, if the dict contains a flat
        ``spike_count`` and ``n_neurons`` alongside a ``cell_type`` key,
        each row is treated individually.
    sim_ms : float
        Total simulation duration in milliseconds.

    Returns
    -------
    dict
        ``{cell_type: {"sim_hz": float, "ref_min": float, "ref_max": float,
        "pass": bool}}`` for each cell type found in both the simulation
        and the reference CSV.
    """
    ref = _load_firing_rate_reference()
    duration_s = sim_ms / 1000.0
    if duration_s <= 0:
        return {}

    results: dict = {}
    for ct, entries in ref.items():
        if ct not in sim:
            continue
        info = sim[ct]
        n = info.get("n_neurons", 1)
        count = info.get("spike_count", 0)
        sim_hz = (count / max(n, 1)) / duration_s

        # Use the broadest range across all conditions
        all_min = min(e[0] for e in entries)
        all_max = max(e[1] for e in entries)

        results[ct] = {
            "sim_hz": round(sim_hz, 4),
            "ref_min": all_min,
            "ref_max": all_max,
            "pass": bool(all_min <= sim_hz <= all_max),
        }
    return results


def validate_sparseness(sim: dict, sim_ms: float) -> dict:
    """Check population sparseness per brain region.

    Reference targets (Honegger et al. 2011):
      - MB: <10% of KCs active per odor presentation
      - AL: 30-60% of PNs active

    Parameters
    ----------
    sim : dict
        Must contain region-level keys with ``{"n_active": int,
        "n_total": int}`` for regions ``"MB"`` and/or ``"AL"``.
    sim_ms : float
        Simulation duration (unused directly, kept for API symmetry).

    Returns
    -------
    dict
        ``{region: {"fraction_active": float, "ref_min": float,
        "ref_max": float, "pass": bool}}``
    """
    targets = {
        "MB": (0.0, 0.10),
        "AL": (0.30, 0.60),
    }
    results: dict = {}
    for region, (lo, hi) in targets.items():
        if region not in sim:
            continue
        info = sim[region]
        total = info.get("n_total", 0)
        if total == 0:
            continue
        frac = info.get("n_active", 0) / total
        results[region] = {
            "fraction_active": round(frac, 4),
            "ref_min": lo,
            "ref_max": hi,
            "pass": bool(lo <= frac <= hi),
        }
    return results


def validate_isi(sim: dict, sim_ms: float) -> dict:
    """Check coefficient of variation of inter-spike intervals.

    Target: CV in [0.5, 1.0] for Poisson-like spiking.

    Parameters
    ----------
    sim : dict
        Must contain ``"cv_values"`` (list of per-neuron CV_ISI floats).
    sim_ms : float
        Simulation duration (unused directly).

    Returns
    -------
    dict
        ``{"mean_cv": float, "ref_min": float, "ref_max": float,
        "pass": bool}``
    """
    cvs = sim.get("cv_values", [])
    if not cvs:
        return {"mean_cv": float("nan"), "ref_min": 0.5, "ref_max": 1.0,
                "pass": False}
    mean_cv = sum(cvs) / len(cvs)
    return {
        "mean_cv": round(mean_cv, 4),
        "ref_min": 0.5,
        "ref_max": 1.0,
        "pass": bool(0.5 <= mean_cv <= 1.0),
    }


def validate_correlation(sim: dict, sim_ms: float) -> dict:
    """Check that within-region spike correlation exceeds between-region.

    Parameters
    ----------
    sim : dict
        Must contain ``"within"`` (float, mean within-region pairwise
        correlation) and ``"between"`` (float, mean between-region
        pairwise correlation).
    sim_ms : float
        Simulation duration (unused directly).

    Returns
    -------
    dict
        ``{"within": float, "between": float, "pass": bool}``
    """
    within = sim.get("within", float("nan"))
    between = sim.get("between", float("nan"))
    passed = False
    if math.isfinite(within) and math.isfinite(between):
        passed = within > between
    return {
        "within": round(within, 4) if math.isfinite(within) else None,
        "between": round(between, 4) if math.isfinite(between) else None,
        "pass": bool(passed),
    }


# ===================================================================
# 4. Plot functions
# ===================================================================

def plot_firing_rate_panel(result: dict, ax: plt.Axes) -> None:
    """Bar chart of simulated vs reference firing rate ranges.

    Parameters
    ----------
    result : dict
        Output of :func:`validate_firing_rates`.
    ax : matplotlib Axes
        Target axes for drawing.
    """
    if not result:
        ax.text(0.5, 0.5, "No firing rate data", ha="center", va="center",
                transform=ax.transAxes)
        return

    types = sorted(result.keys())
    x = list(range(len(types)))
    sim_vals = [result[t]["sim_hz"] for t in types]
    ref_mids = [(result[t]["ref_min"] + result[t]["ref_max"]) / 2.0
                for t in types]
    ref_errs = [(result[t]["ref_max"] - result[t]["ref_min"]) / 2.0
                for t in types]

    bar_w = 0.35
    ax.bar([xi - bar_w / 2 for xi in x], sim_vals, bar_w,
           color=SIM_COLOR, label="Sim")
    ax.bar([xi + bar_w / 2 for xi in x], ref_mids, bar_w,
           yerr=ref_errs, color=REF_COLOR, label="Ref range",
           capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(types)
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title("Firing rates by cell type")
    ax.legend(fontsize=8)


def plot_sparseness_panel(result: dict, ax: plt.Axes) -> None:
    """Bar chart of population sparseness per region.

    Parameters
    ----------
    result : dict
        Output of :func:`validate_sparseness`.
    ax : matplotlib Axes
        Target axes.
    """
    if not result:
        ax.text(0.5, 0.5, "No sparseness data", ha="center", va="center",
                transform=ax.transAxes)
        return

    regions = sorted(result.keys())
    x = list(range(len(regions)))
    sim_vals = [result[r]["fraction_active"] for r in regions]
    ref_mids = [(result[r]["ref_min"] + result[r]["ref_max"]) / 2.0
                for r in regions]
    ref_errs = [(result[r]["ref_max"] - result[r]["ref_min"]) / 2.0
                for r in regions]

    bar_w = 0.35
    ax.bar([xi - bar_w / 2 for xi in x], sim_vals, bar_w,
           color=SIM_COLOR, label="Sim")
    ax.bar([xi + bar_w / 2 for xi in x], ref_mids, bar_w,
           yerr=ref_errs, color=REF_COLOR, label="Ref range",
           capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(regions)
    ax.set_ylabel("Fraction active")
    ax.set_title("Population sparseness")
    ax.legend(fontsize=8)


def plot_isi_panel(result: dict, ax: plt.Axes) -> None:
    """Histogram of CV(ISI) values with reference band.

    Parameters
    ----------
    result : dict
        Output of :func:`validate_isi`.  If the original ``sim`` dict
        contained a ``"cv_values"`` list, pass that as
        ``result["_cv_values"]`` for the full histogram; otherwise a
        single vertical line at ``mean_cv`` is drawn.
    ax : matplotlib Axes
        Target axes.
    """
    cv_vals = result.get("_cv_values", [])
    mean_cv = result.get("mean_cv", float("nan"))

    # Reference band
    ax.axvspan(result.get("ref_min", 0.5), result.get("ref_max", 1.0),
               alpha=0.2, color=REF_COLOR, label="Ref range")

    if cv_vals:
        ax.hist(cv_vals, bins=30, color=SIM_COLOR, alpha=0.7,
                edgecolor="white", label="Per-neuron CV")
    if math.isfinite(mean_cv):
        ax.axvline(mean_cv, color=SIM_COLOR, linewidth=2, linestyle="--",
                   label=f"Mean CV = {mean_cv:.2f}")

    ax.set_xlabel("CV(ISI)")
    ax.set_ylabel("Count")
    ax.set_title("ISI coefficient of variation")
    ax.legend(fontsize=8)


def plot_correlation_panel(result: dict, ax: plt.Axes) -> None:
    """Within vs between region correlation bar chart.

    Parameters
    ----------
    result : dict
        Output of :func:`validate_correlation`.
    ax : matplotlib Axes
        Target axes.
    """
    within = result.get("within")
    between = result.get("between")

    labels = []
    vals = []
    colors = []
    if within is not None:
        labels.append("Within")
        vals.append(within)
        colors.append(SIM_COLOR)
    if between is not None:
        labels.append("Between")
        vals.append(between)
        colors.append(REF_COLOR)

    if not vals:
        ax.text(0.5, 0.5, "No correlation data", ha="center", va="center",
                transform=ax.transAxes)
        return

    ax.bar(labels, vals, color=colors)
    ax.set_ylabel("Mean pairwise correlation")
    ax.set_title("Spike correlation structure")
    status = "PASS" if result.get("pass") else "FAIL"
    ax.text(0.95, 0.95, status, transform=ax.transAxes,
            ha="right", va="top", fontsize=10,
            color="green" if result.get("pass") else "red",
            fontweight="bold")


# ===================================================================
# 5. Summary printer
# ===================================================================

def print_summary(**results: dict) -> None:
    """Print an ASCII table of all validation results.

    Each keyword argument is the name of a validator and its result
    dict.  For example::

        print_summary(
            firing_rates=fr_result,
            sparseness=sp_result,
            isi=isi_result,
            correlation=corr_result,
        )
    """
    header = f"{'Module':<20} {'Metric':<22} {'Value':>10} {'Range':>16} {'Result':>6}"
    sep = "-" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)

    for module, data in results.items():
        if data is None:
            continue

        # Firing rates: nested by cell type
        if module == "firing_rates" and isinstance(data, dict):
            for ct, info in sorted(data.items()):
                val_str = f"{info.get('sim_hz', '?'):>10.3f}" if isinstance(
                    info.get("sim_hz"), (int, float)) else f"{'?':>10}"
                rng = f"[{info.get('ref_min', '?')}, {info.get('ref_max', '?')}]"
                tag = "PASS" if info.get("pass") else "FAIL"
                print(f"{module:<20} {ct:<22} {val_str} {rng:>16} {tag:>6}")
            continue

        # Sparseness: nested by region
        if module == "sparseness" and isinstance(data, dict):
            for region, info in sorted(data.items()):
                val_str = f"{info.get('fraction_active', '?'):>10.4f}" if isinstance(
                    info.get("fraction_active"), (int, float)) else f"{'?':>10}"
                rng = f"[{info.get('ref_min', '?')}, {info.get('ref_max', '?')}]"
                tag = "PASS" if info.get("pass") else "FAIL"
                print(f"{module:<20} {region:<22} {val_str} {rng:>16} {tag:>6}")
            continue

        # ISI: single row
        if module == "isi" and isinstance(data, dict):
            val = data.get("mean_cv", float("nan"))
            val_str = f"{val:>10.4f}" if math.isfinite(val) else f"{'N/A':>10}"
            rng = f"[{data.get('ref_min', '?')}, {data.get('ref_max', '?')}]"
            tag = "PASS" if data.get("pass") else "FAIL"
            print(f"{module:<20} {'mean_cv':<22} {val_str} {rng:>16} {tag:>6}")
            continue

        # Correlation: single row
        if module == "correlation" and isinstance(data, dict):
            w = data.get("within")
            b = data.get("between")
            val_str = f"{w:>10.4f}" if w is not None else f"{'N/A':>10}"
            rng = f"within>between"
            tag = "PASS" if data.get("pass") else "FAIL"
            print(f"{module:<20} {'within_vs_between':<22} {val_str} {rng:>16} {tag:>6}")
            if b is not None:
                print(f"{'':>20} {'between':<22} {b:>10.4f} {'':>16} {'':>6}")
            continue

        # Fallback: try to print whatever dict shape we got
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, dict) and "pass" in v:
                    tag = "PASS" if v["pass"] else "FAIL"
                    print(f"{module:<20} {str(k):<22} {'':>10} {'':>16} {tag:>6}")

    print(sep)
    print()


# ===================================================================
# 6. Module detection
# ===================================================================

def detect_modules(sim: dict) -> dict:
    """Detect which validators can run based on available data.

    Parameters
    ----------
    sim : dict
        Loaded simulation output (e.g. from ``load_csv`` on
        ``metrics.csv``).  Column names determine availability.

    Returns
    -------
    dict
        ``{module_name: bool}`` indicating which validators have
        sufficient data.
    """
    cols = set(sim.keys()) if isinstance(sim, dict) else set()

    has_spikes = "spike_count" in cols
    has_correlation = "correlation" in cols

    # Firing rates need per-type spike data, which comes from a
    # pre-computed summary, not directly from metrics.csv columns.
    # We flag it as available when spike_count is present (the caller
    # must supply the per-type breakdown separately).
    can_firing_rates = has_spikes
    can_sparseness = has_spikes
    can_isi = has_spikes
    can_correlation = has_correlation

    return {
        "firing_rates": can_firing_rates,
        "sparseness": can_sparseness,
        "isi": can_isi,
        "correlation": can_correlation,
    }
