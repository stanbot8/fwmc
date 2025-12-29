#!/usr/bin/env python3
"""Import FlyWire connectome data into FWMC binary format.

Downloads neuron coordinates and synapse tables from the FlyWire CAVE API
or Codex (codex.flywire.ai) HTTP API, converts to compact binary format
for the C++ simulation.

Requirements:
    pip install caveclient cloud-volume numpy pandas

Usage:
    # CAVE API (requires caveclient auth)
    python3 scripts/import_connectome.py                    # full brain
    python3 scripts/import_connectome.py --region MB        # mushroom body only
    python3 scripts/import_connectome.py --region AL        # antennal lobe only
    python3 scripts/import_connectome.py --max-neurons 1000 # subset for testing

    # Codex HTTP API (no auth needed, uses public datasets)
    python3 scripts/import_connectome.py --codex --region MB
    python3 scripts/import_connectome.py --codex --codex-dataset flywire_fafb

    # Test circuit (offline, no API access needed)
    python3 scripts/import_connectome.py --test --test-size 500

    # Utilities
    python3 scripts/import_connectome.py --region MB --cell-types --nt-predictions --validate
    python3 scripts/import_connectome.py --merge data/AL data/MB data/LH --output data/merged
"""

import argparse
import collections
import json
import os
import struct
import sys

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# CellType enum values matching src/core/experiment_config.h
CELL_TYPE_ENUM = {
    "generic":            0,
    "kenyon_cell":        1,  "kc": 1,
    "mbon_cholinergic":   2,
    "mbon_gabaergic":     3,
    "mbon_glutamatergic": 4,
    "dan_ppl1":           5,
    "dan_pam":            6,
    "pn_excitatory":      7,  "epn": 7,
    "pn_inhibitory":      8,  "ipn": 8,
    "ln_local":           9,  "ln": 9,
    "orn":                10,
    "fast_spiking":       11,
    "bursting":           12,
}

NT_MAP = {"ach": 0, "gaba": 1, "glut": 2, "da": 3, "ser": 4, "oct": 5}

REGION_BOUNDS = {
    # (x_min, x_max, y_min, y_max, z_min, z_max) in 4nm voxels
    "MB":   (100000, 140000, 40000, 80000, 2000, 5000),
    "AL":   ( 60000, 100000, 55000, 85000, 1500, 4000),
    "CX":   (120000, 170000, 60000, 90000, 2500, 5500),
    "LH":   ( 85000, 120000, 30000, 60000, 2000, 5000),
    "OL":   ( 30000,  80000, 20000, 70000, 1000, 6000),
    "SEZ":  (100000, 150000, 85000, 120000, 2000, 5000),
    "MBON": (100000, 140000, 40000, 80000, 2000, 5000),
}

# Default NT type for each cell type when generating test data
_TEST_NT = {0: 0, 1: 0, 2: 0, 3: 1, 4: 2, 5: 3, 6: 3, 7: 0, 8: 1, 9: 1, 10: 0, 11: 0, 12: 0}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _map_cell_type_string(s):
    """Map a FlyWire cell_type annotation string to a CellType enum value."""
    s = str(s).strip().lower().replace(" ", "_").replace("-", "_")
    # Direct lookup
    if s in CELL_TYPE_ENUM:
        return CELL_TYPE_ENUM[s]
    # Partial match heuristics based on common FlyWire annotations
    if "kenyon" in s or s.startswith("kc"):
        return 1
    if "mbon" in s:
        if "gaba" in s:
            return 3
        if "glut" in s:
            return 4
        return 2  # default cholinergic MBON
    if "dan" in s or "dopamin" in s:
        if "ppl1" in s:
            return 5
        return 6  # default PAM
    if "pn" in s or "projection" in s:
        if "inhibit" in s or "ipn" in s:
            return 8
        return 7
    if "ln" in s or "local" in s:
        return 9
    if "orn" in s or "receptor" in s:
        return 10
    return 0  # generic fallback


def _read_neurons_bin(path):
    """Read neurons.bin and return (root_ids, positions, types)."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<I", f.read(4))
        root_ids = np.empty(n, dtype=np.uint64)
        positions = np.empty((n, 3), dtype=np.float32)
        types = np.empty(n, dtype=np.uint8)
        for i in range(n):
            rid, x, y, z, t = struct.unpack("<Q3fB", f.read(8 + 12 + 1))
            root_ids[i] = rid
            positions[i] = (x, y, z)
            types[i] = t
    return root_ids, positions, types


def _read_synapses_bin(path):
    """Read synapses.bin and return (pre, post, weight, nt) arrays."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<I", f.read(4))
        pre = np.empty(n, dtype=np.uint32)
        post = np.empty(n, dtype=np.uint32)
        weight = np.empty(n, dtype=np.float32)
        nt = np.empty(n, dtype=np.uint8)
        for i in range(n):
            p, q, w, t = struct.unpack("<IIfB", f.read(4 + 4 + 4 + 1))
            pre[i] = p
            post[i] = q
            weight[i] = w
            nt[i] = t
    return pre, post, weight, nt


def _read_meta(path):
    with open(path, "r") as f:
        return json.load(f)


def _write_neurons_bin(path, root_ids, positions, types):
    n = len(root_ids)
    with open(path, "wb") as f:
        f.write(struct.pack("<I", n))
        for i in range(n):
            f.write(struct.pack("<Q3fB",
                                int(root_ids[i]),
                                float(positions[i, 0]),
                                float(positions[i, 1]),
                                float(positions[i, 2]),
                                int(types[i])))


def _write_synapses_bin(path, pre, post, weight, nt):
    n = len(pre)
    with open(path, "wb") as f:
        f.write(struct.pack("<I", n))
        for i in range(n):
            f.write(struct.pack("<IIfB",
                                int(pre[i]),
                                int(post[i]),
                                float(weight[i]),
                                int(nt[i])))


# ---------------------------------------------------------------------------
# CAVE import
# ---------------------------------------------------------------------------

def import_from_cave(region=None, max_neurons=None, output_dir="data",
                     region_file=None, cell_types=False, nt_predictions=False,
                     subsample=None, validate=False):
    """Import connectome from FlyWire CAVE tables."""
    try:
        from caveclient import CAVEclient
    except ImportError:
        print("Install caveclient: pip install caveclient")
        sys.exit(1)

    client = CAVEclient("flywire_fafb_production_mar2024")
    print(f"Connected to FlyWire (materialization v{client.materialize.version})")

    # Get neuron table
    print("Fetching neurons...")
    neurons = client.materialize.query_table("nuclei_v1")
    print(f"  {len(neurons)} neurons total")

    if region:
        region_upper = region.upper()
        if region_upper not in REGION_BOUNDS:
            print(f"  Warning: unknown region '{region}'. Known: {list(REGION_BOUNDS.keys())}")
            print(f"  Proceeding without spatial filter.")
        else:
            bx = REGION_BOUNDS[region_upper]
            pos = neurons["pt_position"].values
            mask = np.array([
                bx[0] <= p[0] <= bx[1] and
                bx[2] <= p[1] <= bx[3] and
                bx[4] <= p[2] <= bx[5]
                for p in pos
            ])
            neurons = neurons[mask]
            print(f"  {len(neurons)} neurons in {region_upper} bounding box")

    if max_neurons:
        neurons = neurons.head(max_neurons)
        print(f"  Limited to {max_neurons} neurons")

    n_neurons = len(neurons)
    root_ids = neurons["pt_root_id"].values

    # Build root_id to index mapping
    id_to_idx = {rid: i for i, rid in enumerate(root_ids)}

    # ---- Cell type annotations (--cell-types) ----
    type_array = np.zeros(n_neurons, dtype=np.uint8)
    if cell_types:
        print("Fetching cell type annotations...")
        try:
            ct_table = client.materialize.query_table("cell_type_v1")
            ct_map = {}
            for _, row in ct_table.iterrows():
                rid = int(row["pt_root_id"])
                ct_str = str(row.get("cell_type", "generic"))
                ct_map[rid] = _map_cell_type_string(ct_str)
            matched = 0
            for rid, idx in id_to_idx.items():
                if rid in ct_map:
                    type_array[idx] = ct_map[rid]
                    matched += 1
            print(f"  Matched {matched}/{n_neurons} neurons to cell types")
        except Exception as e:
            print(f"  Warning: could not fetch cell types: {e}")

    # ---- Region file (--region-file) ----
    region_labels = {}
    if region_file:
        print(f"Loading region assignments from {region_file}...")
        with open(region_file, "r") as f:
            raw = json.load(f)
        # raw maps root_id (as string key) -> region name
        for rid_str, rname in raw.items():
            rid = int(rid_str)
            if rid in id_to_idx:
                region_labels[id_to_idx[rid]] = rname
        print(f"  Assigned {len(region_labels)} neurons to regions")

    # Get synapse table
    print("Fetching synapses...")
    synapses = client.materialize.query_table("synapses_nt_v1")
    print(f"  {len(synapses)} synapses total")

    # Filter to our neuron set
    mask = (synapses["pre_pt_root_id"].isin(root_ids) &
            synapses["post_pt_root_id"].isin(root_ids))
    synapses = synapses[mask]
    print(f"  {len(synapses)} synapses in subgraph")

    # ---- Subsample (--subsample) ----
    if subsample and subsample < n_neurons:
        print(f"Subsampling to {subsample} neurons...")
        keep_idx = set(np.random.choice(n_neurons, subsample, replace=False).tolist())
        old_to_new = {}
        new_i = 0
        for old_i in sorted(keep_idx):
            old_to_new[old_i] = new_i
            new_i += 1
        # Rebuild neuron data
        keep_sorted = sorted(keep_idx)
        root_ids = root_ids[keep_sorted]
        type_array = type_array[keep_sorted]
        new_region_labels = {}
        for old_i in keep_sorted:
            if old_i in region_labels:
                new_region_labels[old_to_new[old_i]] = region_labels[old_i]
        region_labels = new_region_labels
        # Rebuild id_to_idx
        old_id_to_idx = id_to_idx
        id_to_idx = {rid: i for i, rid in enumerate(root_ids)}
        # Filter synapses: both pre and post must map to kept neurons
        new_keep_rids = set(root_ids.tolist())
        mask = (synapses["pre_pt_root_id"].isin(new_keep_rids) &
                synapses["post_pt_root_id"].isin(new_keep_rids))
        synapses = synapses[mask]
        n_neurons = subsample
        print(f"  {n_neurons} neurons, {len(synapses)} synapses after subsample")

    # ---- NT prediction weights (--nt-predictions) ----
    nt_confidence = {}
    if nt_predictions:
        # FlyWire provides per-synapse NT prediction scores as columns
        # like nts_ach, nts_gaba, etc. Use the max confidence as a weight multiplier.
        nt_score_cols = [c for c in synapses.columns if c.startswith("nts_")]
        if nt_score_cols:
            print(f"  Using NT prediction scores from columns: {nt_score_cols}")
        else:
            print("  Warning: no NT prediction score columns found (nts_*)")

    os.makedirs(output_dir, exist_ok=True)

    # Reconstruct positions from original neurons dataframe
    neurons_df = neurons
    if subsample and subsample < len(neurons_df):
        # positions are already reindexed via keep_sorted above
        pass

    # Write neurons binary: [root_id(u64), x(f32), y(f32), z(f32), type(u8)]
    neuron_path = os.path.join(output_dir, "neurons.bin")
    with open(neuron_path, "wb") as f:
        f.write(struct.pack("<I", n_neurons))
        for idx, (_, row) in enumerate(neurons_df.iterrows()):
            if subsample and idx >= n_neurons:
                break
            pos = row["pt_position"]
            f.write(struct.pack("<Q3fB",
                                int(row["pt_root_id"]),
                                float(pos[0]), float(pos[1]), float(pos[2]),
                                int(type_array[idx]) if idx < len(type_array) else 0))
    print(f"  Wrote {neuron_path} ({n_neurons} neurons)")

    # Write synapses binary: [pre(u32), post(u32), weight(f32), nt(u8)]
    synapse_path = os.path.join(output_dir, "synapses.bin")
    n_syn = 0
    with open(synapse_path, "wb") as f:
        f.write(struct.pack("<I", 0))  # placeholder count
        for _, row in synapses.iterrows():
            pre_id = int(row["pre_pt_root_id"])
            post_id = int(row["post_pt_root_id"])
            if pre_id not in id_to_idx or post_id not in id_to_idx:
                continue
            pre_idx = id_to_idx[pre_id]
            post_idx = id_to_idx[post_id]
            # Skip self-synapses
            if pre_idx == post_idx:
                continue
            nt_str = str(row.get("nt_type", "ach")).lower()
            nt = NT_MAP.get(nt_str, 255)
            nts = float(row.get("nts", 1))  # synapse count/weight

            # Apply NT prediction confidence as weight multiplier
            if nt_predictions:
                nt_score_cols = [c for c in row.index if c.startswith("nts_")]
                if nt_score_cols:
                    scores = [float(row[c]) for c in nt_score_cols if not np.isnan(row[c])]
                    if scores:
                        nts *= max(scores)

            f.write(struct.pack("<IIfB", pre_idx, post_idx, nts, nt))
            n_syn += 1
        # Write actual count
        f.seek(0)
        f.write(struct.pack("<I", n_syn))
    print(f"  Wrote {synapse_path} ({n_syn} synapses)")

    # Write metadata
    meta = {
        "n_neurons": n_neurons,
        "n_synapses": n_syn,
        "region": region or "full_brain",
        "source": "flywire_fafb_production_mar2024",
    }
    if region_labels:
        region_counts = collections.Counter(region_labels.values())
        meta["regions"] = dict(region_counts)
    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Wrote {meta_path}")

    # Write region labels if present
    if region_labels:
        rl_path = os.path.join(output_dir, "region_labels.json")
        with open(rl_path, "w") as f:
            json.dump({str(k): v for k, v in region_labels.items()}, f, indent=2)
        print(f"  Wrote {rl_path}")

    if validate:
        run_validation(output_dir)


# ---------------------------------------------------------------------------
# Multi-region test circuit generation
# ---------------------------------------------------------------------------

def generate_test_circuit(n=100, output_dir="data"):
    """Generate a multi-region test circuit for development without CAVE access.

    Regions:
      AL  (Antennal Lobe)  ~30% - ORN + PN + LN
      MB  (Mushroom Body)  ~50% - KC + MBON + DAN
      LH  (Lateral Horn)   ~20% - PN + LN

    Connectivity:
      - Within-region: ~8% connection probability
      - Between-region: ~1.5% connection probability (inter-region projections)
      - AL->MB projections via PNs
      - AL->LH projections via PNs
      - MB->LH feedback via MBONs
    """
    n = min(n, 10000)
    os.makedirs(output_dir, exist_ok=True)

    np.random.seed(42)

    # Region assignments
    n_al = int(n * 0.30)
    n_mb = int(n * 0.50)
    n_lh = n - n_al - n_mb

    region_names = (["AL"] * n_al) + (["MB"] * n_mb) + (["LH"] * n_lh)

    # Positions: cluster by region with some spread
    positions = np.zeros((n, 3), dtype=np.float32)
    # AL cluster centred at (-50, 0, 0)
    positions[:n_al] = np.random.randn(n_al, 3).astype(np.float32) * 20 + np.array([-50, 0, 0])
    # MB cluster centred at (0, 30, 0)
    al_end = n_al
    mb_end = n_al + n_mb
    positions[al_end:mb_end] = np.random.randn(n_mb, 3).astype(np.float32) * 25 + np.array([0, 30, 0])
    # LH cluster centred at (40, -20, 0)
    positions[mb_end:] = np.random.randn(n_lh, 3).astype(np.float32) * 15 + np.array([40, -20, 0])

    # Cell type assignments per region
    cell_types = np.zeros(n, dtype=np.uint8)
    # AL: ORN (10), PN_excitatory (7), LN_local (9)
    al_idx = np.arange(0, n_al)
    np.random.shuffle(al_idx)
    al_orn_end = int(n_al * 0.4)
    al_pn_end = al_orn_end + int(n_al * 0.35)
    cell_types[al_idx[:al_orn_end]] = 10       # ORN
    cell_types[al_idx[al_orn_end:al_pn_end]] = 7  # PN excitatory
    cell_types[al_idx[al_pn_end:]] = 9          # LN local

    # MB: KC (1), MBON_cholinergic (2), DAN_PAM (6)
    mb_idx = np.arange(n_al, n_al + n_mb)
    np.random.shuffle(mb_idx)
    mb_kc_end = int(n_mb * 0.7)
    mb_mbon_end = mb_kc_end + int(n_mb * 0.15)
    cell_types[mb_idx[:mb_kc_end]] = 1          # KC
    cell_types[mb_idx[mb_kc_end:mb_mbon_end]] = 2  # MBON cholinergic
    cell_types[mb_idx[mb_mbon_end:]] = 6         # DAN PAM

    # LH: PN_excitatory (7), LN_local (9), PN_inhibitory (8)
    lh_idx = np.arange(n_al + n_mb, n)
    np.random.shuffle(lh_idx)
    lh_pn_end = int(n_lh * 0.4)
    lh_ln_end = lh_pn_end + int(n_lh * 0.35)
    cell_types[lh_idx[:lh_pn_end]] = 7          # PN excitatory
    cell_types[lh_idx[lh_pn_end:lh_ln_end]] = 9  # LN local
    cell_types[lh_idx[lh_ln_end:]] = 8           # PN inhibitory

    # Write neurons
    with open(os.path.join(output_dir, "neurons.bin"), "wb") as f:
        f.write(struct.pack("<I", n))
        for i in range(n):
            f.write(struct.pack("<Q3fB",
                                i + 1,  # fake root_id
                                positions[i, 0], positions[i, 1], positions[i, 2],
                                int(cell_types[i])))

    # Build connectivity with higher within-region, lower between-region
    # Also add directed inter-region projection pathways
    region_of = np.array([0] * n_al + [1] * n_mb + [2] * n_lh)
    # Probability matrix: [from_region][to_region]
    # 0=AL, 1=MB, 2=LH
    p_connect = np.array([
        [0.08, 0.03, 0.02],   # AL -> AL(high), AL->MB(PN projection), AL->LH
        [0.005, 0.08, 0.015],  # MB -> AL(low), MB->MB(high), MB->LH(MBON feedback)
        [0.005, 0.005, 0.08],  # LH -> (mostly local)
    ])

    n_syn = 0
    synapse_path = os.path.join(output_dir, "synapses.bin")
    with open(synapse_path, "wb") as f:
        f.write(struct.pack("<I", 0))  # placeholder
        for i in range(n):
            ri = region_of[i]
            ct_i = cell_types[i]
            # Vectorised: compute connection probabilities for all targets
            probs = np.array([p_connect[ri, region_of[j]] for j in range(n)])
            probs[i] = 0  # no self-synapse
            targets = np.where(np.random.rand(n) < probs)[0]
            for j in targets:
                # NT type based on pre-synaptic cell type
                nt = _TEST_NT.get(ct_i, 0)
                w = np.random.exponential(2.0)
                # Stronger weights for within-region
                if region_of[j] == ri:
                    w *= 1.5
                f.write(struct.pack("<IIfB", i, int(j), float(w), nt))
                n_syn += 1
        f.seek(0)
        f.write(struct.pack("<I", n_syn))

    # Region labels
    region_labels = {str(i): region_names[i] for i in range(n)}

    meta = {
        "n_neurons": n,
        "n_synapses": n_syn,
        "region": "multi_region_test",
        "source": "generated",
        "regions": {
            "AL": n_al,
            "MB": n_mb,
            "LH": n_lh,
        },
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(output_dir, "region_labels.json"), "w") as f:
        json.dump(region_labels, f, indent=2)

    print(f"Generated multi-region test circuit: {n} neurons, {n_syn} synapses")
    print(f"  AL: {n_al} neurons | MB: {n_mb} neurons | LH: {n_lh} neurons")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def run_validation(output_dir="data"):
    """Run validation checks on an imported connectome and print a report."""
    print("\n=== Validation ===")
    neuron_path = os.path.join(output_dir, "neurons.bin")
    synapse_path = os.path.join(output_dir, "synapses.bin")

    if not os.path.exists(neuron_path) or not os.path.exists(synapse_path):
        print("  ERROR: neurons.bin or synapses.bin not found in", output_dir)
        return

    root_ids, positions, types = _read_neurons_bin(neuron_path)
    pre, post, weight, nt = _read_synapses_bin(synapse_path)
    n_neurons = len(root_ids)
    n_syn = len(pre)

    report = {"n_neurons": n_neurons, "n_synapses": n_syn, "errors": [], "warnings": []}

    # 1. Self-synapse check
    self_syn = int(np.sum(pre == post))
    if self_syn > 0:
        report["errors"].append(f"{self_syn} self-synapses found")
        print(f"  FAIL: {self_syn} self-synapses")
    else:
        print(f"  OK: no self-synapses")
    report["self_synapses"] = self_syn

    # 2. Index bounds check
    if n_syn > 0:
        max_pre = int(np.max(pre))
        max_post = int(np.max(post))
        oob = int(np.sum(pre >= n_neurons) + np.sum(post >= n_neurons))
        if oob > 0:
            report["errors"].append(f"{oob} out-of-bounds indices (max_pre={max_pre}, max_post={max_post}, n={n_neurons})")
            print(f"  FAIL: {oob} out-of-bounds indices")
        else:
            print(f"  OK: all indices in bounds [0, {n_neurons})")
        report["max_pre_idx"] = max_pre
        report["max_post_idx"] = max_post
    else:
        print("  WARNING: no synapses")
        report["warnings"].append("no synapses")

    # 3. Weight distribution
    if n_syn > 0:
        w_stats = {
            "min": float(np.min(weight)),
            "max": float(np.max(weight)),
            "mean": float(np.mean(weight)),
            "median": float(np.median(weight)),
            "std": float(np.std(weight)),
        }
        neg_w = int(np.sum(weight < 0))
        zero_w = int(np.sum(weight == 0))
        if neg_w > 0:
            report["warnings"].append(f"{neg_w} negative weights")
        if zero_w > 0:
            report["warnings"].append(f"{zero_w} zero weights")
        report["weight_stats"] = w_stats
        print(f"  Weights: min={w_stats['min']:.3f} max={w_stats['max']:.3f} "
              f"mean={w_stats['mean']:.3f} std={w_stats['std']:.3f}")

    # 4. Connected component analysis (undirected)
    print("  Computing connected components...")
    parent = list(range(n_neurons))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[a] = b

    if n_syn > 0:
        for i in range(n_syn):
            union(int(pre[i]), int(post[i]))
        comp_ids = [find(i) for i in range(n_neurons)]
        comp_counts = collections.Counter(comp_ids)
        n_components = len(comp_counts)
        largest = max(comp_counts.values())
        isolated = sum(1 for c in comp_counts.values() if c == 1)
        report["components"] = {
            "total": n_components,
            "largest": largest,
            "isolated_neurons": isolated,
        }
        print(f"  Components: {n_components} total, largest={largest}, isolated={isolated}")
    else:
        report["components"] = {"total": n_neurons, "largest": 1, "isolated_neurons": n_neurons}
        print(f"  Components: {n_neurons} (all isolated, no synapses)")

    # 5. NT type distribution
    if n_syn > 0:
        nt_names = {v: k for k, v in NT_MAP.items()}
        nt_names[255] = "unknown"
        nt_counts = collections.Counter(int(x) for x in nt)
        nt_dist = {nt_names.get(k, f"nt_{k}"): v for k, v in sorted(nt_counts.items())}
        report["nt_distribution"] = nt_dist
        print(f"  NT types: {nt_dist}")

    # 6. Cell type distribution
    type_counts = collections.Counter(int(x) for x in types)
    ct_names = {v: k for k, v in CELL_TYPE_ENUM.items() if k == k.replace(" ", "")}
    # deduplicate: pick the longest name for each value
    ct_name_map = {}
    for k, v in CELL_TYPE_ENUM.items():
        if v not in ct_name_map or len(k) > len(ct_name_map[v]):
            ct_name_map[v] = k
    type_dist = {ct_name_map.get(k, f"type_{k}"): v for k, v in sorted(type_counts.items())}
    report["cell_type_distribution"] = type_dist
    print(f"  Cell types: {type_dist}")

    # Summary
    report["valid"] = len(report["errors"]) == 0
    if report["valid"]:
        print("  RESULT: VALID")
    else:
        print(f"  RESULT: INVALID ({len(report['errors'])} errors)")

    val_path = os.path.join(output_dir, "validation.json")
    with open(val_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Wrote {val_path}")
    return report


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_connectomes(input_dirs, output_dir="data/merged"):
    """Merge multiple region imports into a single connectome.

    Concatenates neuron arrays and remaps synapse indices so they reference
    the merged neuron list.
    """
    os.makedirs(output_dir, exist_ok=True)

    all_root_ids = []
    all_positions = []
    all_types = []
    all_pre = []
    all_post = []
    all_weight = []
    all_nt = []
    all_region_labels = {}
    region_info = {}
    offset = 0

    for d in input_dirs:
        neuron_path = os.path.join(d, "neurons.bin")
        synapse_path = os.path.join(d, "synapses.bin")
        if not os.path.exists(neuron_path):
            print(f"  WARNING: skipping {d} (no neurons.bin)")
            continue

        rids, pos, types = _read_neurons_bin(neuron_path)
        n_local = len(rids)

        all_root_ids.append(rids)
        all_positions.append(pos)
        all_types.append(types)

        # Load region labels if present
        rl_path = os.path.join(d, "region_labels.json")
        if os.path.exists(rl_path):
            with open(rl_path, "r") as f:
                rl = json.load(f)
            for k, v in rl.items():
                all_region_labels[str(int(k) + offset)] = v

        # Load and remap synapses
        if os.path.exists(synapse_path):
            pre, post, weight, nt = _read_synapses_bin(synapse_path)
            all_pre.append(pre + offset)
            all_post.append(post + offset)
            all_weight.append(weight)
            all_nt.append(nt)

        # Try to get region name from meta
        meta_path = os.path.join(d, "meta.json")
        region_name = os.path.basename(d)
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            region_name = meta.get("region", region_name)
        region_info[region_name] = n_local
        # Assign region labels for neurons without them
        for i in range(n_local):
            key = str(offset + i)
            if key not in all_region_labels:
                all_region_labels[key] = region_name

        print(f"  Loaded {d}: {n_local} neurons, region={region_name}")
        offset += n_local

    # Concatenate
    merged_rids = np.concatenate(all_root_ids)
    merged_pos = np.concatenate(all_positions)
    merged_types = np.concatenate(all_types)

    _write_neurons_bin(os.path.join(output_dir, "neurons.bin"),
                       merged_rids, merged_pos, merged_types)

    if all_pre:
        merged_pre = np.concatenate(all_pre)
        merged_post = np.concatenate(all_post)
        merged_weight = np.concatenate(all_weight)
        merged_nt = np.concatenate(all_nt)
    else:
        merged_pre = np.array([], dtype=np.uint32)
        merged_post = np.array([], dtype=np.uint32)
        merged_weight = np.array([], dtype=np.float32)
        merged_nt = np.array([], dtype=np.uint8)

    _write_synapses_bin(os.path.join(output_dir, "synapses.bin"),
                        merged_pre, merged_post, merged_weight, merged_nt)

    n_total = len(merged_rids)
    n_syn = len(merged_pre)
    meta = {
        "n_neurons": n_total,
        "n_synapses": n_syn,
        "region": "merged",
        "source": "merged",
        "regions": region_info,
        "input_dirs": input_dirs,
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(output_dir, "region_labels.json"), "w") as f:
        json.dump(all_region_labels, f, indent=2)

    print(f"Merged: {n_total} neurons, {n_syn} synapses from {len(input_dirs)} sources")
    print(f"  Regions: {region_info}")


# ---------------------------------------------------------------------------
# Codex HTTP API import (no auth required)
# ---------------------------------------------------------------------------

def import_from_codex(region=None, max_neurons=None, output_dir="data",
                      dataset="flywire_fafb", validate=False):
    """Import connectome from FlyWire Codex public HTTP API.

    Codex (codex.flywire.ai) provides a public REST API for querying the
    FlyWire connectome without requiring CAVE authentication. This is the
    easiest path for users who just want to run a simulation.

    The API docs are at: https://codex.flywire.ai/api/docs
    """
    try:
        import urllib.request
    except ImportError:
        print("ERROR: urllib is required (standard library)")
        sys.exit(1)

    base_url = f"https://codex.flywire.ai/api/v1/{dataset}"
    os.makedirs(output_dir, exist_ok=True)

    # Fetch neurons
    print(f"Fetching neurons from Codex ({dataset})...")
    params = []
    if region:
        params.append(f"brain_region={region.upper()}")
    if max_neurons:
        params.append(f"limit={max_neurons}")
    else:
        params.append("limit=10000")  # default cap to avoid huge downloads

    neuron_url = f"{base_url}/neurons?{'&'.join(params)}"
    print(f"  GET {neuron_url}")

    try:
        req = urllib.request.Request(neuron_url)
        req.add_header("Accept", "application/json")
        with urllib.request.urlopen(req, timeout=120) as resp:
            neuron_data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"  ERROR: Failed to fetch neurons: {e}")
        print("  Hint: check your internet connection and try --test for offline mode")
        sys.exit(1)

    neurons_list = neuron_data if isinstance(neuron_data, list) else neuron_data.get("data", [])
    n_neurons = len(neurons_list)
    print(f"  {n_neurons} neurons fetched")

    if n_neurons == 0:
        print("  ERROR: no neurons returned. Try a different region or increase limit.")
        sys.exit(1)

    # Build index mapping
    root_ids = np.array([int(n.get("root_id", n.get("id", i))) for i, n in enumerate(neurons_list)], dtype=np.uint64)
    id_to_idx = {int(rid): i for i, rid in enumerate(root_ids)}

    positions = np.zeros((n_neurons, 3), dtype=np.float32)
    types = np.zeros(n_neurons, dtype=np.uint8)
    for i, n in enumerate(neurons_list):
        pos = n.get("position", n.get("pt_position", [0, 0, 0]))
        if isinstance(pos, dict):
            positions[i] = [pos.get("x", 0), pos.get("y", 0), pos.get("z", 0)]
        else:
            positions[i] = [float(pos[0]), float(pos[1]), float(pos[2])]
        ct_str = str(n.get("cell_type", n.get("type", "generic")))
        types[i] = _map_cell_type_string(ct_str)

    # Write neurons
    _write_neurons_bin(os.path.join(output_dir, "neurons.bin"),
                       root_ids, positions, types)
    print(f"  Wrote neurons.bin ({n_neurons} neurons)")

    # Fetch synapses
    print("Fetching synapses from Codex...")
    synapse_params = []
    if region:
        synapse_params.append(f"brain_region={region.upper()}")
    synapse_params.append(f"limit={min(n_neurons * 50, 500000)}")

    synapse_url = f"{base_url}/synapses?{'&'.join(synapse_params)}"
    print(f"  GET {synapse_url}")

    try:
        req = urllib.request.Request(synapse_url)
        req.add_header("Accept", "application/json")
        with urllib.request.urlopen(req, timeout=300) as resp:
            synapse_data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"  WARNING: Failed to fetch synapses: {e}")
        print("  Writing empty synapse file")
        synapse_data = []

    syn_list = synapse_data if isinstance(synapse_data, list) else synapse_data.get("data", [])

    pre_arr, post_arr, w_arr, nt_arr = [], [], [], []
    for s in syn_list:
        pre_id = int(s.get("pre_root_id", s.get("pre_pt_root_id", -1)))
        post_id = int(s.get("post_root_id", s.get("post_pt_root_id", -1)))
        if pre_id not in id_to_idx or post_id not in id_to_idx:
            continue
        pre_idx = id_to_idx[pre_id]
        post_idx = id_to_idx[post_id]
        if pre_idx == post_idx:
            continue
        nt_str = str(s.get("nt_type", s.get("neuropil", "ach"))).lower()
        nt = NT_MAP.get(nt_str, 255)
        w = float(s.get("syn_count", s.get("nts", s.get("weight", 1.0))))
        pre_arr.append(pre_idx)
        post_arr.append(post_idx)
        w_arr.append(w)
        nt_arr.append(nt)

    n_syn = len(pre_arr)
    _write_synapses_bin(os.path.join(output_dir, "synapses.bin"),
                        np.array(pre_arr, dtype=np.uint32),
                        np.array(post_arr, dtype=np.uint32),
                        np.array(w_arr, dtype=np.float32),
                        np.array(nt_arr, dtype=np.uint8))
    print(f"  Wrote synapses.bin ({n_syn} synapses)")

    # Metadata
    meta = {
        "n_neurons": n_neurons,
        "n_synapses": n_syn,
        "region": region or "full_brain",
        "source": f"codex_{dataset}",
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Wrote meta.json")

    # Cell type distribution
    type_counts = collections.Counter(int(x) for x in types)
    ct_name_map = {}
    for k, v in CELL_TYPE_ENUM.items():
        if v not in ct_name_map or len(k) > len(ct_name_map[v]):
            ct_name_map[v] = k
    type_dist = {ct_name_map.get(k, f"type_{k}"): v for k, v in sorted(type_counts.items())}
    print(f"  Cell types: {type_dist}")

    if validate:
        run_validation(output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Import FlyWire connectome")
    parser.add_argument("--region", type=str, default=None,
                        help="Brain region (MB, AL, CX, etc.)")
    parser.add_argument("--max-neurons", type=int, default=None)
    parser.add_argument("--output", type=str, default="data")
    parser.add_argument("--test", action="store_true",
                        help="Generate test circuit (no CAVE needed)")
    parser.add_argument("--test-size", type=int, default=100,
                        help="Number of neurons in test circuit (max 10000)")

    # Codex API
    parser.add_argument("--codex", action="store_true",
                        help="Use Codex HTTP API (no auth needed) instead of CAVE")
    parser.add_argument("--codex-dataset", type=str, default="flywire_fafb",
                        help="Codex dataset name (default: flywire_fafb)")

    # New flags
    parser.add_argument("--region-file", type=str, default=None,
                        help="JSON file mapping root_id -> region name")
    parser.add_argument("--cell-types", action="store_true",
                        help="Download cell type annotations from FlyWire cell_type_v1 table")
    parser.add_argument("--nt-predictions", action="store_true",
                        help="Use NT prediction confidence scores as weight multipliers")
    parser.add_argument("--subsample", type=int, default=None,
                        help="Randomly subsample to N neurons, preserving connectivity")
    parser.add_argument("--validate", action="store_true",
                        help="Run validation checks after import")
    parser.add_argument("--merge", nargs="+", metavar="DIR",
                        help="Merge multiple region directories into one connectome")

    args = parser.parse_args()

    if args.merge:
        merge_connectomes(args.merge, args.output)
        if args.validate:
            run_validation(args.output)
    elif args.codex:
        import_from_codex(
            region=args.region,
            max_neurons=args.max_neurons,
            output_dir=args.output,
            dataset=args.codex_dataset,
            validate=args.validate,
        )
    elif args.test:
        generate_test_circuit(args.test_size, args.output)
        if args.validate:
            run_validation(args.output)
    else:
        import_from_cave(
            region=args.region,
            max_neurons=args.max_neurons,
            output_dir=args.output,
            region_file=args.region_file,
            cell_types=args.cell_types,
            nt_predictions=args.nt_predictions,
            subsample=args.subsample,
            validate=args.validate,
        )


if __name__ == "__main__":
    main()
