#!/usr/bin/env python3
"""Live dashboard for FWMC spiking neural network simulator.

Watches simulation output files and streams live visualization to a browser
via Server-Sent Events. Renders using Canvas 2D API (no external JS deps).

Usage:
    python3 scripts/live_dashboard.py --data results/ --port 8050
    python3 scripts/live_dashboard.py --data results/ --duration 10000 --port 8050

The dashboard polls the output directory for growing spikes.bin / voltages.bin
files and streams new frames to the browser for real-time rendering.
"""

import argparse
import http.server
import json
import os
import struct
import sys
import threading
import time
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np

# ---------------------------------------------------------------------------
# Binary file watchers
# ---------------------------------------------------------------------------

class SpikeWatcher:
    """Incrementally reads spikes.bin as it grows."""

    def __init__(self, path):
        self.path = path
        self.n_neurons = 0
        self.n_steps_read = 0
        self.header_read = False
        self._file = None
        self._file_pos = 0

    def poll(self, max_frames=100):
        """Read any new frames appended since last poll.

        Returns list of (time_ms, spiked_array) tuples.
        """
        if not os.path.exists(self.path):
            return []

        try:
            f = open(self.path, "rb")
        except OSError:
            return []

        try:
            f.seek(0, 2)
            file_size = f.tell()

            if not self.header_read:
                if file_size < 8:
                    f.close()
                    return []
                f.seek(0)
                self.n_neurons, _ = struct.unpack("<II", f.read(8))
                if self.n_neurons == 0:
                    f.close()
                    return []
                self.header_read = True
                self._file_pos = 8

            frame_size = 4 + self.n_neurons  # f32 time + u8*n_neurons
            f.seek(self._file_pos)
            frames = []
            count = 0
            while count < max_frames:
                data = f.read(frame_size)
                if len(data) < frame_size:
                    break
                t = struct.unpack("<f", data[:4])[0]
                spiked = np.frombuffer(data[4:], dtype=np.uint8).copy()
                frames.append((t, spiked))
                self._file_pos += frame_size
                self.n_steps_read += 1
                count += 1

            f.close()
            return frames
        except Exception:
            f.close()
            return []


class VoltageWatcher:
    """Incrementally reads voltages.bin as it grows."""

    def __init__(self, path):
        self.path = path
        self.n_neurons = 0
        self.header_read = False
        self._file_pos = 0

    def poll(self, max_frames=100):
        if not os.path.exists(self.path):
            return []

        try:
            f = open(self.path, "rb")
        except OSError:
            return []

        try:
            f.seek(0, 2)
            file_size = f.tell()

            if not self.header_read:
                if file_size < 8:
                    f.close()
                    return []
                f.seek(0)
                self.n_neurons, _ = struct.unpack("<II", f.read(8))
                if self.n_neurons == 0:
                    f.close()
                    return []
                self.header_read = True
                self._file_pos = 8

            frame_size = 4 + self.n_neurons * 4
            f.seek(self._file_pos)
            frames = []
            count = 0
            while count < max_frames:
                data = f.read(frame_size)
                if len(data) < frame_size:
                    break
                t = struct.unpack("<f", data[:4])[0]
                v = np.frombuffer(data[4:], dtype=np.float32).copy()
                frames.append((t, v))
                self._file_pos += frame_size
                count += 1

            f.close()
            return frames
        except Exception:
            f.close()
            return []


# ---------------------------------------------------------------------------
# Data aggregator: runs in background thread
# ---------------------------------------------------------------------------

class DashboardState:
    """Thread-safe container for current dashboard state."""

    def __init__(self, raster_window_ms=500.0, rate_history_s=5.0):
        self.lock = threading.Lock()
        self.raster_window_ms = raster_window_ms
        self.rate_history_s = rate_history_s

        # Current simulation state
        self.sim_time_ms = 0.0
        self.n_neurons = 0
        self.total_spikes = 0
        self.steps_processed = 0
        self.wall_start = time.time()

        # Raster buffer: list of (time, list_of_firing_neuron_indices)
        self.raster_events = []

        # Population rate history: list of (time, spike_count)
        self.rate_history = []

        # Neuron state counts
        self.n_spiking = 0
        self.n_resting = 0
        self.n_refractory = 0  # approximate: recently spiked

        # Recent spike times per neuron for refractory estimate
        self._last_spike = None

    def update(self, frames):
        """Process new spike frames."""
        with self.lock:
            for t, spiked in frames:
                self.sim_time_ms = t
                self.steps_processed += 1

                if self.n_neurons == 0:
                    self.n_neurons = len(spiked)
                    self._last_spike = np.full(self.n_neurons, -1e9)

                firing = np.where(spiked > 0)[0]
                n_firing = len(firing)
                self.total_spikes += n_firing

                # Raster events (keep within window)
                self.raster_events.append((float(t), firing.tolist()))
                cutoff = t - self.raster_window_ms
                while self.raster_events and self.raster_events[0][0] < cutoff:
                    self.raster_events.pop(0)

                # Rate history
                self.rate_history.append((float(t), int(n_firing)))
                cutoff_rate = t - self.rate_history_s * 1000.0
                while self.rate_history and self.rate_history[0][0] < cutoff_rate:
                    self.rate_history.pop(0)

                # Neuron state classification
                self._last_spike[firing] = t
                self.n_spiking = n_firing
                refractory_mask = (t - self._last_spike) < 2.0  # 2ms refractory
                refractory_mask[firing] = False  # don't double-count
                self.n_refractory = int(refractory_mask.sum())
                self.n_resting = self.n_neurons - self.n_spiking - self.n_refractory

    def snapshot(self):
        """Return JSON-serializable snapshot of current state."""
        with self.lock:
            elapsed = time.time() - self.wall_start
            rt_ratio = (self.sim_time_ms / 1000.0) / elapsed if elapsed > 0 else 0

            # Downsample raster for transmission (max 200 events)
            raster = self.raster_events[-200:] if len(self.raster_events) > 200 \
                else list(self.raster_events)

            # Downsample rate history (max 500 points)
            rate = self.rate_history[-500:] if len(self.rate_history) > 500 \
                else list(self.rate_history)

            return {
                "sim_time_ms": round(self.sim_time_ms, 2),
                "n_neurons": self.n_neurons,
                "total_spikes": self.total_spikes,
                "steps": self.steps_processed,
                "rt_ratio": round(rt_ratio, 3),
                "wall_s": round(elapsed, 1),
                "raster": raster,
                "rate": rate,
                "neuron_states": {
                    "spiking": self.n_spiking,
                    "refractory": self.n_refractory,
                    "resting": self.n_resting,
                },
            }


def poller_thread(spike_watcher, state, poll_interval=0.1):
    """Background thread that polls spike file and updates state."""
    while True:
        frames = spike_watcher.poll(max_frames=500)
        if frames:
            state.update(frames)
        time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# HTML / JS dashboard (embedded)
# ---------------------------------------------------------------------------

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FWMC Live Dashboard</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
    background: #0d1117; color: #c9d1d9;
}
.header {
    background: #161b22; border-bottom: 1px solid #30363d;
    padding: 12px 20px; display: flex; align-items: center; gap: 16px;
}
.header h1 { font-size: 16px; color: #58a6ff; font-weight: 600; }
.status { font-size: 12px; color: #8b949e; }
.status .live { color: #3fb950; }
.stats-bar {
    display: flex; gap: 24px; padding: 10px 20px;
    background: #161b22; border-bottom: 1px solid #30363d;
    font-size: 13px;
}
.stat { display: flex; flex-direction: column; }
.stat-label { color: #8b949e; font-size: 10px; text-transform: uppercase; }
.stat-value { color: #f0f6fc; font-size: 18px; font-weight: 600; font-variant-numeric: tabular-nums; }
.grid {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 1px; background: #30363d; padding: 1px;
}
.panel {
    background: #0d1117; padding: 12px;
    display: flex; flex-direction: column;
}
.panel-title {
    font-size: 11px; color: #8b949e; text-transform: uppercase;
    letter-spacing: 0.5px; margin-bottom: 8px;
}
canvas { width: 100%; background: #0d1117; border-radius: 4px; }
.full-width { grid-column: 1 / -1; }
</style>
</head>
<body>
<div class="header">
    <h1>FWMC Live Dashboard</h1>
    <div class="status"><span class="live" id="conn-status">&#9679;</span> Connecting...</div>
</div>
<div class="stats-bar">
    <div class="stat"><span class="stat-label">Sim Time</span><span class="stat-value" id="sim-time">0.0 ms</span></div>
    <div class="stat"><span class="stat-label">Neurons</span><span class="stat-value" id="n-neurons">0</span></div>
    <div class="stat"><span class="stat-label">Total Spikes</span><span class="stat-value" id="total-spikes">0</span></div>
    <div class="stat"><span class="stat-label">Steps</span><span class="stat-value" id="steps">0</span></div>
    <div class="stat"><span class="stat-label">Realtime Ratio</span><span class="stat-value" id="rt-ratio">0x</span></div>
    <div class="stat"><span class="stat-label">Wall Time</span><span class="stat-value" id="wall-time">0s</span></div>
</div>
<div class="grid">
    <div class="panel full-width">
        <div class="panel-title">Spike Raster (rolling 500ms)</div>
        <canvas id="raster" height="250"></canvas>
    </div>
    <div class="panel">
        <div class="panel-title">Population Firing Rate (5s history)</div>
        <canvas id="rate-chart" height="200"></canvas>
    </div>
    <div class="panel">
        <div class="panel-title">Neuron State</div>
        <canvas id="state-chart" height="200"></canvas>
    </div>
</div>

<script>
// ---- State ----
let latestData = null;
let connected = false;

// ---- SSE Connection ----
function connect() {
    const es = new EventSource('/events');
    const statusEl = document.getElementById('conn-status');
    const statusParent = statusEl.parentElement;

    es.onopen = () => {
        connected = true;
        statusParent.innerHTML = '<span class="live">&#9679;</span> Live';
    };
    es.onmessage = (e) => {
        try { latestData = JSON.parse(e.data); } catch(ex) {}
    };
    es.onerror = () => {
        connected = false;
        statusParent.innerHTML = '<span style="color:#f85149">&#9679;</span> Disconnected, retrying...';
    };
}

// ---- Rendering ----
function getCtx(id) {
    const c = document.getElementById(id);
    const dpr = window.devicePixelRatio || 1;
    const rect = c.getBoundingClientRect();
    c.width = rect.width * dpr;
    c.height = rect.height * dpr;
    const ctx = c.getContext('2d');
    ctx.scale(dpr, dpr);
    return { ctx, w: rect.width, h: rect.height };
}

function drawRaster(data) {
    const { ctx, w, h } = getCtx('raster');
    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0, 0, w, h);

    const raster = data.raster;
    if (!raster || raster.length === 0) return;

    const nNeurons = data.n_neurons || 1;
    const times = raster.map(r => r[0]);
    const tMin = Math.min(...times);
    const tMax = Math.max(...times);
    const tRange = Math.max(tMax - tMin, 1);

    const pad = { l: 50, r: 10, t: 10, b: 25 };
    const plotW = w - pad.l - pad.r;
    const plotH = h - pad.t - pad.b;

    // Grid lines
    ctx.strokeStyle = '#21262d';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
        const y = pad.t + (plotH * i / 4);
        ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
    }

    // Spike dots
    ctx.fillStyle = '#58a6ff';
    for (const [t, neurons] of raster) {
        const x = pad.l + ((t - tMin) / tRange) * plotW;
        for (const nIdx of neurons) {
            const y = pad.t + plotH - (nIdx / nNeurons) * plotH;
            ctx.fillRect(x - 0.5, y - 0.5, 1.5, 1.5);
        }
    }

    // Axes labels
    ctx.fillStyle = '#8b949e';
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(tMin.toFixed(0) + ' ms', pad.l, h - 5);
    ctx.fillText(tMax.toFixed(0) + ' ms', w - pad.r, h - 5);
    ctx.textAlign = 'right';
    ctx.fillText('0', pad.l - 4, h - pad.b);
    ctx.fillText(nNeurons.toString(), pad.l - 4, pad.t + 10);
}

function drawRate(data) {
    const { ctx, w, h } = getCtx('rate-chart');
    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0, 0, w, h);

    const rate = data.rate;
    if (!rate || rate.length < 2) return;

    const pad = { l: 50, r: 10, t: 10, b: 25 };
    const plotW = w - pad.l - pad.r;
    const plotH = h - pad.t - pad.b;

    const times = rate.map(r => r[0]);
    const counts = rate.map(r => r[1]);
    const tMin = times[0];
    const tMax = times[times.length - 1];
    const tRange = Math.max(tMax - tMin, 1);
    const maxCount = Math.max(...counts, 1);

    // Grid
    ctx.strokeStyle = '#21262d';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
        const y = pad.t + (plotH * i / 4);
        ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
    }

    // Line
    ctx.beginPath();
    ctx.strokeStyle = '#3fb950';
    ctx.lineWidth = 1.5;
    for (let i = 0; i < rate.length; i++) {
        const x = pad.l + ((times[i] - tMin) / tRange) * plotW;
        const y = pad.t + plotH - (counts[i] / maxCount) * plotH;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Fill under curve
    const lastX = pad.l + plotW;
    ctx.lineTo(lastX, pad.t + plotH);
    ctx.lineTo(pad.l, pad.t + plotH);
    ctx.closePath();
    ctx.fillStyle = 'rgba(63, 185, 80, 0.1)';
    ctx.fill();

    // Labels
    ctx.fillStyle = '#8b949e';
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(tMin.toFixed(0), pad.l, h - 5);
    ctx.fillText(tMax.toFixed(0) + ' ms', w - pad.r, h - 5);
    ctx.textAlign = 'right';
    ctx.fillText('0', pad.l - 4, pad.t + plotH);
    ctx.fillText(maxCount.toString(), pad.l - 4, pad.t + 10);
}

function drawStates(data) {
    const { ctx, w, h } = getCtx('state-chart');
    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0, 0, w, h);

    const states = data.neuron_states;
    if (!states) return;

    const bars = [
        { label: 'Spiking', value: states.spiking || 0, color: '#f85149' },
        { label: 'Refractory', value: states.refractory || 0, color: '#d29922' },
        { label: 'Resting', value: states.resting || 0, color: '#3fb950' },
    ];

    const total = bars.reduce((s, b) => s + b.value, 0) || 1;
    const pad = { l: 80, r: 40, t: 20, b: 20 };
    const plotW = w - pad.l - pad.r;
    const plotH = h - pad.t - pad.b;
    const barH = Math.min(40, plotH / bars.length - 10);
    const gap = (plotH - barH * bars.length) / (bars.length + 1);

    for (let i = 0; i < bars.length; i++) {
        const b = bars[i];
        const y = pad.t + gap * (i + 1) + barH * i;
        const bw = (b.value / total) * plotW;

        // Bar background
        ctx.fillStyle = '#21262d';
        ctx.fillRect(pad.l, y, plotW, barH);

        // Bar fill
        ctx.fillStyle = b.color;
        ctx.fillRect(pad.l, y, bw, barH);

        // Label
        ctx.fillStyle = '#c9d1d9';
        ctx.font = '12px monospace';
        ctx.textAlign = 'right';
        ctx.fillText(b.label, pad.l - 8, y + barH / 2 + 4);

        // Value
        ctx.fillStyle = '#f0f6fc';
        ctx.textAlign = 'left';
        ctx.fillText(b.value.toLocaleString(), pad.l + bw + 6, y + barH / 2 + 4);
    }
}

function updateStats(data) {
    document.getElementById('sim-time').textContent = data.sim_time_ms.toFixed(1) + ' ms';
    document.getElementById('n-neurons').textContent = data.n_neurons.toLocaleString();
    document.getElementById('total-spikes').textContent = data.total_spikes.toLocaleString();
    document.getElementById('steps').textContent = data.steps.toLocaleString();
    document.getElementById('rt-ratio').textContent = data.rt_ratio.toFixed(2) + 'x';
    document.getElementById('wall-time').textContent = data.wall_s.toFixed(0) + 's';
}

function render() {
    if (latestData) {
        updateStats(latestData);
        drawRaster(latestData);
        drawRate(latestData);
        drawStates(latestData);
    }
    requestAnimationFrame(render);
}

// ---- Init ----
connect();
requestAnimationFrame(render);

// Handle resize
window.addEventListener('resize', () => {
    if (latestData) {
        drawRaster(latestData);
        drawRate(latestData);
        drawStates(latestData);
    }
});
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# HTTP Server with SSE
# ---------------------------------------------------------------------------

class DashboardHandler(http.server.BaseHTTPRequestHandler):
    """Serves the dashboard HTML and SSE event stream."""

    # Class-level references set before server starts
    dashboard_state = None

    def log_message(self, format, *args):
        # Suppress default request logging
        pass

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/" or parsed.path == "/index.html":
            self._serve_html()
        elif parsed.path == "/events":
            self._serve_sse()
        elif parsed.path == "/snapshot":
            self._serve_json()
        else:
            self.send_error(404)

    def _serve_html(self):
        content = DASHBOARD_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(content)

    def _serve_json(self):
        snapshot = self.dashboard_state.snapshot()
        body = json.dumps(snapshot).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _serve_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        try:
            while True:
                snapshot = self.dashboard_state.snapshot()
                payload = json.dumps(snapshot, separators=(",", ":"))
                message = f"data: {payload}\n\n"
                self.wfile.write(message.encode("utf-8"))
                self.wfile.flush()
                time.sleep(0.1)  # 10 Hz update rate
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass


class ThreadedHTTPServer(http.server.HTTPServer):
    """HTTP server that handles each request in a new thread."""
    allow_reuse_address = True
    daemon_threads = True

    def process_request(self, request, client_address):
        t = threading.Thread(target=self.process_request_thread,
                             args=(request, client_address), daemon=True)
        t.start()

    def process_request_thread(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FWMC Live Dashboard: real-time spiking network visualization"
    )
    parser.add_argument("--data", "-d", default="results",
                        help="Directory containing spikes.bin output (default: results)")
    parser.add_argument("--port", "-p", type=int, default=8050,
                        help="HTTP server port (default: 8050)")
    parser.add_argument("--duration", type=float, default=0,
                        help="Expected sim duration in ms (informational, default: 0)")
    parser.add_argument("--poll-interval", type=float, default=0.1,
                        help="File poll interval in seconds (default: 0.1)")
    parser.add_argument("--raster-window", type=float, default=500.0,
                        help="Raster display window in ms (default: 500)")
    parser.add_argument("--rate-history", type=float, default=5.0,
                        help="Rate chart history in seconds (default: 5)")
    args = parser.parse_args()

    data_dir = args.data
    spike_path = os.path.join(data_dir, "spikes.bin")

    print("=" * 60)
    print("  FWMC Live Dashboard")
    print("=" * 60)
    print(f"  Data directory  : {os.path.abspath(data_dir)}")
    print(f"  Spike file      : {spike_path}")
    print(f"  Poll interval   : {args.poll_interval}s")
    print(f"  Raster window   : {args.raster_window} ms")
    print(f"  Rate history    : {args.rate_history} s")
    print(f"  Server port     : {args.port}")
    if args.duration > 0:
        print(f"  Expected dur.   : {args.duration} ms")
    print()

    if not os.path.isdir(data_dir):
        print(f"Warning: data directory '{data_dir}' does not exist yet.")
        print("  The dashboard will start and wait for files to appear.")
        print()

    # Initialize state and watcher
    state = DashboardState(
        raster_window_ms=args.raster_window,
        rate_history_s=args.rate_history,
    )
    watcher = SpikeWatcher(spike_path)

    # Start poller thread
    poller = threading.Thread(
        target=poller_thread,
        args=(watcher, state, args.poll_interval),
        daemon=True,
    )
    poller.start()

    # Start HTTP server
    DashboardHandler.dashboard_state = state
    server = ThreadedHTTPServer(("0.0.0.0", args.port), DashboardHandler)

    url = f"http://localhost:{args.port}"
    print(f"  Dashboard running at: {url}")
    print(f"  Press Ctrl+C to stop")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
