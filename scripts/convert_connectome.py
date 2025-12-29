#!/usr/bin/env python3
"""Convert between connectome formats for the FWMC pipeline.

Supported conversions:
    CSV          -> FWMC binary (neurons.bin + synapses.bin)
    EdgeList     -> FWMC binary
    GraphML      -> FWMC binary
    NeuPrint     -> FWMC binary
    FWMC binary  -> GraphML / GEXF (for Gephi visualisation)

Usage:
    python3 scripts/convert_connectome.py --input data.csv --format csv --output data/
    python3 scripts/convert_connectome.py --input edges.txt --format edgelist --output data/
    python3 scripts/convert_connectome.py --input brain.graphml --format graphml --output data/
    python3 scripts/convert_connectome.py --input neuprint.json --format neuprint --output data/
    python3 scripts/convert_connectome.py --input data/ --format fwmc --export graphml --output brain.graphml
    python3 scripts/convert_connectome.py --input data/ --format fwmc --export gexf --output brain.gexf
"""

import argparse
import collections
import json
import os
import struct
import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom

import numpy as np


# ---------------------------------------------------------------------------
# NT / cell-type constants (shared with import_connectome.py)
# ---------------------------------------------------------------------------

NT_MAP = {"ach": 0, "gaba": 1, "glut": 2, "da": 3, "ser": 4, "oct": 5}
NT_NAMES = {v: k for k, v in NT_MAP.items()}
NT_NAMES[255] = "unknown"


# ---------------------------------------------------------------------------
# Binary I/O helpers
# ---------------------------------------------------------------------------

def _write_neurons_bin(path, root_ids, positions, types):
    n = len(root_ids)
    with open(path, "wb") as f:
        f.write(struct.pack("<I", n))
        for i in range(n):
            x, y, z = float(positions[i][0]), float(positions[i][1]), float(positions[i][2])
            f.write(struct.pack("<Q3fB", int(root_ids[i]), x, y, z, int(types[i])))
    return n


def _write_synapses_bin(path, pre, post, weight, nt):
    n = len(pre)
    with open(path, "wb") as f:
        f.write(struct.pack("<I", n))
        for i in range(n):
            f.write(struct.pack("<IIfB",
                                int(pre[i]), int(post[i]),
                                float(weight[i]), int(nt[i])))
    return n


def _read_neurons_bin(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<I", f.read(4))
        root_ids = np.empty(n, dtype=np.uint64)
        positions = np.empty((n, 3), dtype=np.float32)
        types = np.empty(n, dtype=np.uint8)
        for i in range(n):
            rid, x, y, z, t = struct.unpack("<Q3fB", f.read(21))
            root_ids[i] = rid
            positions[i] = (x, y, z)
            types[i] = t
    return root_ids, positions, types


def _read_synapses_bin(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<I", f.read(4))
        pre = np.empty(n, dtype=np.uint32)
        post = np.empty(n, dtype=np.uint32)
        weight = np.empty(n, dtype=np.float32)
        nt = np.empty(n, dtype=np.uint8)
        for i in range(n):
            p, q, w, t = struct.unpack("<IIfB", f.read(13))
            pre[i] = p
            post[i] = q
            weight[i] = w
            nt[i] = t
    return pre, post, weight, nt


# ---------------------------------------------------------------------------
# CSV converter
# ---------------------------------------------------------------------------

def convert_csv(input_path, output_dir):
    """Convert CSV with columns: pre_id, post_id, weight, nt_type.

    Optional extra columns: pre_x, pre_y, pre_z, post_x, post_y, post_z, cell_type
    """
    import csv

    print(f"Reading CSV: {input_path}")
    rows = []
    with open(input_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    if not rows:
        print("  ERROR: CSV is empty")
        return

    print(f"  {len(rows)} rows, columns: {fieldnames}")

    # Collect unique neuron IDs
    neuron_ids = set()
    for r in rows:
        neuron_ids.add(int(r["pre_id"]))
        neuron_ids.add(int(r["post_id"]))
    neuron_ids = sorted(neuron_ids)
    id_to_idx = {nid: i for i, nid in enumerate(neuron_ids)}
    n_neurons = len(neuron_ids)

    # Positions (if columns exist, otherwise zeros)
    has_pos = "pre_x" in (fieldnames or [])
    positions = np.zeros((n_neurons, 3), dtype=np.float32)
    types = np.zeros(n_neurons, dtype=np.uint8)

    if has_pos:
        # Assign positions from the first occurrence of each neuron
        assigned = set()
        for r in rows:
            pre = int(r["pre_id"])
            post = int(r["post_id"])
            if pre not in assigned:
                idx = id_to_idx[pre]
                positions[idx] = (float(r.get("pre_x", 0)),
                                  float(r.get("pre_y", 0)),
                                  float(r.get("pre_z", 0)))
                assigned.add(pre)
            if post not in assigned and "post_x" in r:
                idx = id_to_idx[post]
                positions[idx] = (float(r.get("post_x", 0)),
                                  float(r.get("post_y", 0)),
                                  float(r.get("post_z", 0)))
                assigned.add(post)

    # Build synapse arrays
    pre_arr = []
    post_arr = []
    weight_arr = []
    nt_arr = []
    skipped = 0
    for r in rows:
        pre = id_to_idx[int(r["pre_id"])]
        post = id_to_idx[int(r["post_id"])]
        if pre == post:
            skipped += 1
            continue
        w = float(r.get("weight", 1.0))
        nt_str = str(r.get("nt_type", "ach")).strip().lower()
        nt = NT_MAP.get(nt_str, 255)
        pre_arr.append(pre)
        post_arr.append(post)
        weight_arr.append(w)
        nt_arr.append(nt)

    if skipped:
        print(f"  Skipped {skipped} self-synapses")

    os.makedirs(output_dir, exist_ok=True)
    root_ids = np.array(neuron_ids, dtype=np.uint64)
    nn = _write_neurons_bin(os.path.join(output_dir, "neurons.bin"),
                            root_ids, positions, types)
    ns = _write_synapses_bin(os.path.join(output_dir, "synapses.bin"),
                             np.array(pre_arr, dtype=np.uint32),
                             np.array(post_arr, dtype=np.uint32),
                             np.array(weight_arr, dtype=np.float32),
                             np.array(nt_arr, dtype=np.uint8))

    _write_meta(output_dir, nn, ns, "csv", input_path, weight_arr)
    print(f"  Wrote {nn} neurons, {ns} synapses to {output_dir}")


# ---------------------------------------------------------------------------
# EdgeList converter
# ---------------------------------------------------------------------------

def convert_edgelist(input_path, output_dir):
    """Convert space-separated edge list: pre post weight [nt]

    Lines starting with # or % are comments. If only two columns,
    weight defaults to 1.0.
    """
    print(f"Reading edge list: {input_path}")
    edges = []
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            pre = int(parts[0])
            post = int(parts[1])
            w = float(parts[2]) if len(parts) > 2 else 1.0
            nt_str = parts[3].lower() if len(parts) > 3 else "ach"
            edges.append((pre, post, w, nt_str))

    if not edges:
        print("  ERROR: no edges found")
        return

    print(f"  {len(edges)} edges")

    neuron_ids = sorted(set(e[0] for e in edges) | set(e[1] for e in edges))
    id_to_idx = {nid: i for i, nid in enumerate(neuron_ids)}
    n_neurons = len(neuron_ids)

    pre_arr, post_arr, weight_arr, nt_arr = [], [], [], []
    skipped = 0
    for pre, post, w, nt_str in edges:
        pi, qi = id_to_idx[pre], id_to_idx[post]
        if pi == qi:
            skipped += 1
            continue
        pre_arr.append(pi)
        post_arr.append(qi)
        weight_arr.append(w)
        nt_arr.append(NT_MAP.get(nt_str, 255))

    if skipped:
        print(f"  Skipped {skipped} self-synapses")

    os.makedirs(output_dir, exist_ok=True)
    positions = np.zeros((n_neurons, 3), dtype=np.float32)
    types = np.zeros(n_neurons, dtype=np.uint8)
    root_ids = np.array(neuron_ids, dtype=np.uint64)

    nn = _write_neurons_bin(os.path.join(output_dir, "neurons.bin"),
                            root_ids, positions, types)
    ns = _write_synapses_bin(os.path.join(output_dir, "synapses.bin"),
                             np.array(pre_arr, dtype=np.uint32),
                             np.array(post_arr, dtype=np.uint32),
                             np.array(weight_arr, dtype=np.float32),
                             np.array(nt_arr, dtype=np.uint8))

    _write_meta(output_dir, nn, ns, "edgelist", input_path, weight_arr)
    print(f"  Wrote {nn} neurons, {ns} synapses to {output_dir}")


# ---------------------------------------------------------------------------
# GraphML converter
# ---------------------------------------------------------------------------

def convert_graphml(input_path, output_dir):
    """Convert standard GraphML with optional weight and nt attributes on edges."""
    print(f"Reading GraphML: {input_path}")
    tree = ET.parse(input_path)
    root = tree.getroot()

    # Handle namespace
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    # Discover attribute keys
    key_map = {}  # key id -> attr name
    for key_el in root.findall(f"{ns}key"):
        kid = key_el.get("id")
        aname = key_el.get("attr.name", kid)
        key_map[kid] = aname.lower()

    graph = root.find(f"{ns}graph")
    if graph is None:
        print("  ERROR: no <graph> element found")
        return

    # Nodes
    node_ids = []
    node_data = {}  # node_id -> {attr: val}
    for node in graph.findall(f"{ns}node"):
        nid = node.get("id")
        node_ids.append(nid)
        data = {}
        for d in node.findall(f"{ns}data"):
            k = d.get("key")
            attr = key_map.get(k, k)
            data[attr] = d.text
        node_data[nid] = data

    # Map string node IDs to integer indices
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    n_neurons = len(node_ids)
    print(f"  {n_neurons} nodes")

    # Try to get numeric IDs for root_ids
    root_ids = np.arange(1, n_neurons + 1, dtype=np.uint64)
    for i, nid in enumerate(node_ids):
        try:
            root_ids[i] = int(nid)
        except (ValueError, TypeError):
            pass  # keep sequential id

    positions = np.zeros((n_neurons, 3), dtype=np.float32)
    types = np.zeros(n_neurons, dtype=np.uint8)
    for i, nid in enumerate(node_ids):
        d = node_data.get(nid, {})
        positions[i, 0] = float(d.get("x", 0))
        positions[i, 1] = float(d.get("y", 0))
        positions[i, 2] = float(d.get("z", 0))

    # Edges
    pre_arr, post_arr, weight_arr, nt_arr = [], [], [], []
    skipped = 0
    for edge in graph.findall(f"{ns}edge"):
        src = edge.get("source")
        tgt = edge.get("target")
        if src not in id_to_idx or tgt not in id_to_idx:
            continue
        pi, qi = id_to_idx[src], id_to_idx[tgt]
        if pi == qi:
            skipped += 1
            continue
        w = 1.0
        nt_val = 0
        for d in edge.findall(f"{ns}data"):
            k = d.get("key")
            attr = key_map.get(k, k)
            if attr == "weight":
                w = float(d.text)
            elif attr in ("nt", "nt_type", "neurotransmitter"):
                nt_val = NT_MAP.get(d.text.strip().lower(), 255)
        pre_arr.append(pi)
        post_arr.append(qi)
        weight_arr.append(w)
        nt_arr.append(nt_val)

    print(f"  {len(pre_arr)} edges")
    if skipped:
        print(f"  Skipped {skipped} self-loops")

    os.makedirs(output_dir, exist_ok=True)
    nn = _write_neurons_bin(os.path.join(output_dir, "neurons.bin"),
                            root_ids, positions, types)
    ns = _write_synapses_bin(os.path.join(output_dir, "synapses.bin"),
                             np.array(pre_arr, dtype=np.uint32),
                             np.array(post_arr, dtype=np.uint32),
                             np.array(weight_arr, dtype=np.float32),
                             np.array(nt_arr, dtype=np.uint8))

    _write_meta(output_dir, nn, ns, "graphml", input_path, weight_arr)
    print(f"  Wrote {nn} neurons, {ns} synapses to {output_dir}")


# ---------------------------------------------------------------------------
# NeuPrint converter
# ---------------------------------------------------------------------------

def convert_neuprint(input_path, output_dir):
    """Convert NeuPrint JSON export (from neuprint.janelia.org API).

    Expected format (result of a connectivity query):
    {
      "columns": ["bodyId_pre", "bodyId_post", "weight", ...],
      "data": [[pre, post, w, ...], ...]
    }

    Or the simpler adjacency format:
    [
      {"bodyId_pre": 123, "bodyId_post": 456, "weight": 5, "type_pre": "KC", ...},
      ...
    ]
    """
    print(f"Reading NeuPrint JSON: {input_path}")
    with open(input_path, "r") as f:
        raw = json.load(f)

    # Detect format
    edges = []
    if isinstance(raw, dict) and "columns" in raw and "data" in raw:
        cols = raw["columns"]
        # Find relevant column indices
        col_idx = {c: i for i, c in enumerate(cols)}
        pre_col = None
        post_col = None
        weight_col = None
        for c in cols:
            cl = c.lower()
            if "pre" in cl and "body" in cl:
                pre_col = col_idx[c]
            elif "post" in cl and "body" in cl:
                post_col = col_idx[c]
            elif cl == "weight" or cl == "weightHP":
                weight_col = col_idx[c]
        if pre_col is None or post_col is None:
            print(f"  ERROR: could not find pre/post body ID columns. Columns: {cols}")
            return
        for row in raw["data"]:
            pre = int(row[pre_col])
            post = int(row[post_col])
            w = float(row[weight_col]) if weight_col is not None else 1.0
            edges.append((pre, post, w))
    elif isinstance(raw, list):
        for item in raw:
            pre = int(item.get("bodyId_pre", item.get("pre", 0)))
            post = int(item.get("bodyId_post", item.get("post", 0)))
            w = float(item.get("weight", item.get("weightHp", 1.0)))
            edges.append((pre, post, w))
    else:
        print("  ERROR: unrecognized NeuPrint JSON format")
        return

    if not edges:
        print("  ERROR: no edges found")
        return

    print(f"  {len(edges)} edges")

    neuron_ids = sorted(set(e[0] for e in edges) | set(e[1] for e in edges))
    id_to_idx = {nid: i for i, nid in enumerate(neuron_ids)}
    n_neurons = len(neuron_ids)

    pre_arr, post_arr, weight_arr, nt_arr = [], [], [], []
    skipped = 0
    for pre, post, w in edges:
        pi, qi = id_to_idx[pre], id_to_idx[post]
        if pi == qi:
            skipped += 1
            continue
        pre_arr.append(pi)
        post_arr.append(qi)
        weight_arr.append(w)
        nt_arr.append(0)  # NeuPrint doesn't always include NT, default ACh

    if skipped:
        print(f"  Skipped {skipped} self-connections")

    os.makedirs(output_dir, exist_ok=True)
    positions = np.zeros((n_neurons, 3), dtype=np.float32)
    types = np.zeros(n_neurons, dtype=np.uint8)
    root_ids = np.array(neuron_ids, dtype=np.uint64)

    nn = _write_neurons_bin(os.path.join(output_dir, "neurons.bin"),
                            root_ids, positions, types)
    ns_count = _write_synapses_bin(os.path.join(output_dir, "synapses.bin"),
                                   np.array(pre_arr, dtype=np.uint32),
                                   np.array(post_arr, dtype=np.uint32),
                                   np.array(weight_arr, dtype=np.float32),
                                   np.array(nt_arr, dtype=np.uint8))

    _write_meta(output_dir, nn, ns_count, "neuprint", input_path, weight_arr)
    print(f"  Wrote {nn} neurons, {ns_count} synapses to {output_dir}")


# ---------------------------------------------------------------------------
# FWMC binary -> GraphML / GEXF export
# ---------------------------------------------------------------------------

def export_graphml(input_dir, output_path):
    """Export FWMC binary connectome to GraphML format."""
    print(f"Exporting {input_dir} to GraphML: {output_path}")
    root_ids, positions, types = _read_neurons_bin(
        os.path.join(input_dir, "neurons.bin"))
    pre, post, weight, nt = _read_synapses_bin(
        os.path.join(input_dir, "synapses.bin"))

    n_neurons = len(root_ids)
    n_syn = len(pre)

    # Load region labels if available
    region_labels = {}
    rl_path = os.path.join(input_dir, "region_labels.json")
    if os.path.exists(rl_path):
        with open(rl_path, "r") as f:
            region_labels = json.load(f)

    # Build GraphML
    graphml = ET.Element("graphml", xmlns="http://graphml.graphstruct.org/xmlns")

    # Attribute declarations
    for kid, aname, afor, atype, default in [
        ("d0", "root_id", "node", "long", "0"),
        ("d1", "x", "node", "float", "0.0"),
        ("d2", "y", "node", "float", "0.0"),
        ("d3", "z", "node", "float", "0.0"),
        ("d4", "cell_type", "node", "int", "0"),
        ("d5", "region", "node", "string", ""),
        ("d6", "weight", "edge", "float", "1.0"),
        ("d7", "nt_type", "edge", "string", "ach"),
    ]:
        key_el = ET.SubElement(graphml, "key", id=kid,
                               **{"attr.name": aname, "for": afor, "attr.type": atype})
        ET.SubElement(key_el, "default").text = default

    graph = ET.SubElement(graphml, "graph", id="G", edgedefault="directed")

    for i in range(n_neurons):
        node = ET.SubElement(graph, "node", id=str(i))
        ET.SubElement(node, "data", key="d0").text = str(int(root_ids[i]))
        ET.SubElement(node, "data", key="d1").text = f"{positions[i, 0]:.4f}"
        ET.SubElement(node, "data", key="d2").text = f"{positions[i, 1]:.4f}"
        ET.SubElement(node, "data", key="d3").text = f"{positions[i, 2]:.4f}"
        ET.SubElement(node, "data", key="d4").text = str(int(types[i]))
        region = region_labels.get(str(i), "")
        if region:
            ET.SubElement(node, "data", key="d5").text = region

    for i in range(n_syn):
        edge = ET.SubElement(graph, "edge",
                             source=str(int(pre[i])),
                             target=str(int(post[i])))
        ET.SubElement(edge, "data", key="d6").text = f"{weight[i]:.4f}"
        ET.SubElement(edge, "data", key="d7").text = NT_NAMES.get(int(nt[i]), "unknown")

    xml_str = ET.tostring(graphml, encoding="unicode")
    # Pretty print
    dom = minidom.parseString(xml_str)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(dom.toprettyxml(indent="  "))

    print(f"  Wrote {n_neurons} nodes, {n_syn} edges to {output_path}")


def export_gexf(input_dir, output_path):
    """Export FWMC binary connectome to GEXF format (for Gephi)."""
    print(f"Exporting {input_dir} to GEXF: {output_path}")
    root_ids, positions, types = _read_neurons_bin(
        os.path.join(input_dir, "neurons.bin"))
    pre, post, weight, nt = _read_synapses_bin(
        os.path.join(input_dir, "synapses.bin"))

    n_neurons = len(root_ids)
    n_syn = len(pre)

    region_labels = {}
    rl_path = os.path.join(input_dir, "region_labels.json")
    if os.path.exists(rl_path):
        with open(rl_path, "r") as f:
            region_labels = json.load(f)

    # Build GEXF XML
    gexf = ET.Element("gexf")
    gexf.set("xmlns", "http://www.gexf.net/1.3")
    gexf.set("xmlns:viz", "http://www.gexf.net/1.3/viz")
    gexf.set("version", "1.3")
    meta = ET.SubElement(gexf, "meta")
    ET.SubElement(meta, "creator").text = "FWMC convert_connectome.py"
    ET.SubElement(meta, "description").text = "FlyWire connectome export"

    graph = ET.SubElement(gexf, "graph", defaultedgetype="directed", mode="static")

    # Node attributes
    node_attrs = ET.SubElement(graph, "attributes", **{"class": "node"})
    ET.SubElement(node_attrs, "attribute", id="0", title="root_id", type="long")
    ET.SubElement(node_attrs, "attribute", id="1", title="cell_type", type="integer")
    ET.SubElement(node_attrs, "attribute", id="2", title="region", type="string")

    # Edge attributes
    edge_attrs = ET.SubElement(graph, "attributes", **{"class": "edge"})
    ET.SubElement(edge_attrs, "attribute", id="0", title="nt_type", type="string")

    # Nodes
    nodes_el = ET.SubElement(graph, "nodes")
    for i in range(n_neurons):
        node = ET.SubElement(nodes_el, "node", id=str(i),
                             label=f"n{i}")
        attvals = ET.SubElement(node, "attvalues")
        ET.SubElement(attvals, "attvalue", **{"for": "0", "value": str(int(root_ids[i]))})
        ET.SubElement(attvals, "attvalue", **{"for": "1", "value": str(int(types[i]))})
        region = region_labels.get(str(i), "")
        if region:
            ET.SubElement(attvals, "attvalue", **{"for": "2", "value": region})
        # Position (viz namespace)
        pos_el = ET.SubElement(node, "viz:position",
                               x=f"{positions[i, 0]:.2f}",
                               y=f"{positions[i, 1]:.2f}",
                               z=f"{positions[i, 2]:.2f}")

    # Edges
    edges_el = ET.SubElement(graph, "edges")
    for i in range(n_syn):
        edge = ET.SubElement(edges_el, "edge",
                             id=str(i),
                             source=str(int(pre[i])),
                             target=str(int(post[i])),
                             weight=f"{weight[i]:.4f}")
        attvals = ET.SubElement(edge, "attvalues")
        ET.SubElement(attvals, "attvalue",
                      **{"for": "0", "value": NT_NAMES.get(int(nt[i]), "unknown")})

    xml_str = ET.tostring(gexf, encoding="unicode")
    dom = minidom.parseString(xml_str)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(dom.toprettyxml(indent="  "))

    print(f"  Wrote {n_neurons} nodes, {n_syn} edges to {output_path}")


# ---------------------------------------------------------------------------
# Metadata / stats helper
# ---------------------------------------------------------------------------

def _write_meta(output_dir, n_neurons, n_synapses, source_format, source_path, weight_list):
    """Write meta.json and print statistics."""
    w_arr = np.array(weight_list, dtype=np.float32) if weight_list else np.array([])
    stats = {}
    if len(w_arr) > 0:
        stats = {
            "weight_min": float(np.min(w_arr)),
            "weight_max": float(np.max(w_arr)),
            "weight_mean": float(np.mean(w_arr)),
            "weight_std": float(np.std(w_arr)),
        }
        print(f"  Weight stats: min={stats['weight_min']:.3f} max={stats['weight_max']:.3f} "
              f"mean={stats['weight_mean']:.3f} std={stats['weight_std']:.3f}")

    meta = {
        "n_neurons": n_neurons,
        "n_synapses": n_synapses,
        "source_format": source_format,
        "source_path": os.path.basename(source_path),
        "region": "imported",
        "source": f"converted_from_{source_format}",
    }
    meta.update(stats)
    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Wrote {meta_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert between connectome formats for FWMC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Supported input formats:
  csv       - CSV with columns: pre_id, post_id, weight, nt_type
  edgelist  - Space-separated: pre post weight [nt]
  graphml   - Standard GraphML with weight/nt attributes
  neuprint  - JSON from neuprint.janelia.org API
  fwmc      - FWMC binary (for export to graphml/gexf)

Export formats (use with --format fwmc):
  graphml   - GraphML XML
  gexf      - GEXF XML (Gephi)
""")
    parser.add_argument("--input", "-i", required=True,
                        help="Input file or directory")
    parser.add_argument("--format", "-f", required=True,
                        choices=["csv", "edgelist", "graphml", "neuprint", "fwmc"],
                        help="Input format")
    parser.add_argument("--output", "-o", required=True,
                        help="Output directory (for import) or file path (for export)")
    parser.add_argument("--export", "-e", choices=["graphml", "gexf"],
                        help="Export format (only with --format fwmc)")

    args = parser.parse_args()

    if args.format == "fwmc":
        if not args.export:
            print("ERROR: --export is required when --format is fwmc")
            sys.exit(1)
        if args.export == "graphml":
            export_graphml(args.input, args.output)
        elif args.export == "gexf":
            export_gexf(args.input, args.output)
    elif args.format == "csv":
        convert_csv(args.input, args.output)
    elif args.format == "edgelist":
        convert_edgelist(args.input, args.output)
    elif args.format == "graphml":
        convert_graphml(args.input, args.output)
    elif args.format == "neuprint":
        convert_neuprint(args.input, args.output)


if __name__ == "__main__":
    main()
