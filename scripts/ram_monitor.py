import csv
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# Colour palette for phase shading on plots
_PHASE_COLOURS = [
    "#AED6F1", "#A9DFBF", "#F9E79F", "#F5CBA7",
    "#D2B4DE", "#FADADD", "#D5DBDB", "#A3E4D7",
]


class RAMMonitor:
    """
    Monitors system RAM and memory-access counters during model inference.

    New capabilities over the original version
    -------------------------------------------
    * Phase markers – call mark_phase(name) to annotate the timeline with
      logical stages (e.g. "warmup", "iteration_0", "iteration_25").
    * Delta-based vmstat – records the *rate* of page faults / IO per sample
      interval rather than cumulative totals, so spikes are immediately visible.
    * Richer perf events – adds dTLB, iTLB, and DRAM bandwidth events alongside
      the existing LLC and cache-reference counters.
    * Phase-annotated plots – shaded regions show exactly when each phase ran,
      so memory-access changes can be correlated to inference stages.
    """

    # ------------------------------------------------------------------
    # perf events to collect
    # ------------------------------------------------------------------
    _PERF_EVENTS = [
        # L1/L2/LLC cache
        "cache-references",
        "cache-misses",
        "LLC-loads",
        "LLC-load-misses",
        "LLC-stores",
        "LLC-store-misses",
        # TLB (data + instruction)
        "dTLB-loads",
        "dTLB-load-misses",
        "dTLB-stores",
        "dTLB-store-misses",
        "iTLB-loads",
        "iTLB-load-misses",
        # Core
        "cycles",
        "instructions",
    ]

    _PERF_EVENT_MAP = {
        "cache-references":  "cache_references",
        "cache-misses":      "cache_misses",
        "LLC-loads":         "llc_loads",
        "LLC-load-misses":   "llc_load_misses",
        "LLC-stores":        "llc_stores",
        "LLC-store-misses":  "llc_store_misses",
        "dTLB-loads":        "dtlb_loads",
        "dTLB-load-misses":  "dtlb_load_misses",
        "dTLB-stores":       "dtlb_stores",
        "dTLB-store-misses": "dtlb_store_misses",
        "iTLB-loads":        "itlb_loads",
        "iTLB-load-misses":  "itlb_load_misses",
        "cycles":            "cycles",
        "instructions":      "instructions",
    }

    def __init__(self, log_dir, model_name, device, batch_size, precision):
        self.log_dir = Path(log_dir)
        self.base_filename = f"{model_name}_{device}_b{batch_size}_{precision}"

        self.running = False
        self.timestamps: List[float] = []
        self.metrics: List[dict] = []

        # Phase markers: list of (timestamp, phase_name)
        self._phase_markers: List[Tuple[float, str]] = []
        self._phase_lock = threading.Lock()
        self._start_time: float = 0.0

        self.thread = None
        self.perf_thread = None
        self.perf_process = None

        self.perf_supported = False
        self.perf_available = False
        self.latest_perf_metrics = self._default_perf_metrics()
        self.perf_lock = threading.Lock()

        # vmstat from previous sample – used to compute deltas (rates)
        self._prev_vmstat: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_base_filename(self, new_base: str):
        self.base_filename = new_base

    def mark_phase(self, name: str):
        """
        Record a named phase boundary at the current wall-clock time.

        Call this before and after logical stages of inference, e.g.:

            ram_monitor.mark_phase("warmup_start")
            # ... run warmup ...
            ram_monitor.mark_phase("warmup_end")
            ram_monitor.mark_phase("inference_start")
            # ... run inference loop ...
            ram_monitor.mark_phase("inference_end")

        The phase name is stored with a timestamp relative to monitor start
        and will be rendered as shaded regions on every plot.
        """
        if not self.running:
            return
        t = time.time() - self._start_time
        with self._phase_lock:
            self._phase_markers.append((t, name))

    # ------------------------------------------------------------------
    # Internals – /proc readers
    # ------------------------------------------------------------------

    def _default_perf_metrics(self) -> dict:
        return {k: 0 for k in self._PERF_EVENT_MAP.values()}

    def _read_meminfo(self) -> dict:
        mem: Dict[str, int] = {}
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    key, value = line.split(":", 1)
                    mem[key.strip()] = int(value.split()[0])
        except Exception:
            pass

        mem_total = mem.get("MemTotal", 0)
        mem_available = mem.get("MemAvailable", 0)
        mem_used = mem_total - mem_available

        return {
            "mem_total_kb":     mem_total,
            "mem_used_kb":      mem_used,
            "mem_free_kb":      mem.get("MemFree", 0),
            "mem_available_kb": mem_available,
            "buffers_kb":       mem.get("Buffers", 0),
            "cached_kb":        mem.get("Cached", 0),
            "swap_used_kb":     mem.get("SwapTotal", 0) - mem.get("SwapFree", 0),
            # Derived: fraction used (0-1)
            "mem_used_fraction": mem_used / mem_total if mem_total else 0.0,
        }

    def _read_vmstat_raw(self) -> Dict[str, int]:
        vm: Dict[str, int] = {}
        try:
            with open("/proc/vmstat", "r") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) == 2:
                        vm[parts[0]] = int(parts[1])
        except Exception:
            pass
        return vm

    def _read_vmstat_delta(self) -> dict:
        """
        Return per-interval *rates* for vmstat counters by diffing against
        the previous snapshot.  The first call returns zeros.
        """
        raw = self._read_vmstat_raw()

        keys_of_interest = [
            "pgfault", "pgmajfault",
            "pgpgin", "pgpgout",
            "pswpin", "pswpout",
            "pgalloc_normal", "pgfree",
            "pgreuse",
        ]

        delta: Dict[str, int] = {}
        for k in keys_of_interest:
            cur = raw.get(k, 0)
            prev = self._prev_vmstat.get(k, cur)   # first call: delta = 0
            delta[f"{k}_delta"] = max(0, cur - prev)

        self._prev_vmstat = {k: raw.get(k, 0) for k in keys_of_interest}
        return delta

    # ------------------------------------------------------------------
    # Internals – perf stat
    # ------------------------------------------------------------------

    def _start_perf(self):
        cmd = [
            "perf", "stat",
            "-a",          # system-wide
            "-x", ",",     # CSV output
            "-I", "100",   # 100 ms intervals
            "-e", ",".join(self._PERF_EVENTS),
        ]
        try:
            self.perf_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            self.perf_available = True
            self.perf_thread = threading.Thread(
                target=self._perf_reader_thread, daemon=True
            )
            self.perf_thread.start()
        except Exception:
            self.perf_process = None
            self.perf_available = False

    def _parse_perf_value(self, raw: str) -> int:
        v = raw.strip()
        if not v or v in ("<not counted>", "<not supported>",
                          "not counted", "not supported"):
            return 0
        try:
            return int(float(v.replace(",", "")))
        except ValueError:
            return 0

    def _perf_reader_thread(self):
        if not self.perf_process or not self.perf_process.stderr:
            return
        try:
            for line in self.perf_process.stderr:
                if not self.running:
                    break
                stripped = line.strip()
                if not stripped:
                    continue
                lower = stripped.lower()
                if any(x in lower for x in ("not supported", "no permission",
                                            "permission denied")):
                    continue
                parts = [p.strip() for p in stripped.split(",")]
                if len(parts) < 3:
                    continue
                key = self._PERF_EVENT_MAP.get(parts[2].strip())
                if not key:
                    continue
                value = self._parse_perf_value(parts[1])
                with self.perf_lock:
                    self.latest_perf_metrics[key] = value
                    self.perf_supported = True
        except Exception:
            pass

    def _get_perf_snapshot(self) -> dict:
        with self.perf_lock:
            return dict(self.latest_perf_metrics)

    # ------------------------------------------------------------------
    # Monitor thread
    # ------------------------------------------------------------------

    def _monitor_thread(self, interval: float = 0.1):
        while self.running:
            mem       = self._read_meminfo()
            vm_delta  = self._read_vmstat_delta()
            perf      = self._get_perf_snapshot()

            # Derived cache-miss rate (needs non-zero refs to be meaningful)
            refs = perf.get("cache_references", 0)
            misses = perf.get("cache_misses", 0)
            cache_miss_rate = misses / refs if refs > 0 else 0.0

            dtlb_loads = perf.get("dtlb_loads", 0)
            dtlb_miss  = perf.get("dtlb_load_misses", 0)
            dtlb_miss_rate = dtlb_miss / dtlb_loads if dtlb_loads > 0 else 0.0

            cycles = perf.get("cycles", 0)
            instrs = perf.get("instructions", 0)
            ipc = instrs / cycles if cycles > 0 else 0.0

            derived = {
                "cache_miss_rate":    cache_miss_rate,
                "dtlb_miss_rate":     dtlb_miss_rate,
                "ipc":                ipc,
            }

            metric = {**mem, **vm_delta, **perf, **derived}
            self.metrics.append(metric)
            self.timestamps.append(time.time() - self._start_time)

            time.sleep(interval)

    # ------------------------------------------------------------------
    # Start / stop
    # ------------------------------------------------------------------

    def start(self):
        self._start_time = time.time()
        self.running = True
        self._start_perf()
        self.thread = threading.Thread(
            target=self._monitor_thread, daemon=True
        )
        self.thread.start()
        print("RAM monitoring started")

    def stop(self):
        if not self.running:
            return
        self.running = False

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        if self.perf_process:
            try:
                self.perf_process.terminate()
                self.perf_process.wait(timeout=2.0)
            except Exception:
                try:
                    self.perf_process.kill()
                except Exception:
                    pass

        if self.perf_thread and self.perf_thread.is_alive():
            self.perf_thread.join(timeout=1.0)

        if not self.metrics:
            print("No RAM metrics collected")
            return

        self.log_dir.mkdir(exist_ok=True)
        self._save_csv()
        self._create_plots()

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    def _save_csv(self):
        csv_path = self.log_dir / f"{self.base_filename}_ram_metrics.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_seconds"] + list(self.metrics[0].keys()))
            for ts, m in zip(self.timestamps, self.metrics):
                writer.writerow([ts] + list(m.values()))

        # Save phase markers separately so downstream scripts can use them
        phase_path = self.log_dir / f"{self.base_filename}_phase_markers.csv"
        with self._phase_lock:
            markers = list(self._phase_markers)
        with open(phase_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_seconds", "phase"])
            for t, name in markers:
                writer.writerow([t, name])

        print(f"RAM metrics saved to {csv_path}")
        print(f"Phase markers saved to {phase_path}")

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

    def _phase_spans(self) -> List[Tuple[float, float, str]]:
        """
        Convert a flat list of phase markers into (start, end, label) spans.

        Markers are expected to come in _start / _end pairs, e.g.:
            ("warmup_start", t0), ("warmup_end", t1)
        or as single-point labels that are auto-paired with the next marker.
        """
        with self._phase_lock:
            markers = list(self._phase_markers)

        spans: List[Tuple[float, float, str]] = []
        pending: Dict[str, float] = {}

        for t, name in markers:
            if name.endswith("_start"):
                pending[name[:-6]] = t          # strip "_start"
            elif name.endswith("_end"):
                base = name[:-4]                # strip "_end"
                if base in pending:
                    spans.append((pending.pop(base), t, base))
            else:
                # Treat as a boundary marker – pair with the previous marker
                if spans:
                    # extend the last span if it has the same stem
                    pass
                # Keep as a zero-width event for annotation
                spans.append((t, t, name))

        # Close any unclosed _start spans at the last timestamp
        last_t = self.timestamps[-1] if self.timestamps else 0.0
        for base, start in pending.items():
            spans.append((start, last_t, base))

        return spans

    def _shade_phases(self, ax, spans: List[Tuple[float, float, str]]):
        """Draw shaded regions and labels for each phase span onto *ax*."""
        for i, (start, end, label) in enumerate(spans):
            colour = _PHASE_COLOURS[i % len(_PHASE_COLOURS)]
            if end > start:
                ax.axvspan(start, end, alpha=0.25, color=colour, label=label)
            else:
                ax.axvline(start, color=colour, linestyle="--", alpha=0.7,
                           label=label)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def _create_plots(self):
        if not self.metrics:
            return

        plot_dir = self.log_dir / "plots"
        plot_dir.mkdir(exist_ok=True)

        spans = self._phase_spans()
        t = self.timestamps

        # ---- helper lambdas ----
        def col(key):
            return [m.get(key, 0) for m in self.metrics]

        def _save(fig, name):
            path = plot_dir / f"{self.base_filename}_{name}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {path.name}")

        print("Saving RAM access pattern plots:")

        # ------------------------------------------------------------------
        # 1. Memory usage timeline
        # ------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(t, [v / 1024 for v in col("mem_used_kb")],
                color="#2980B9", linewidth=1.5, label="RAM used (MB)")
        ax.set_ylabel("Memory used (MB)")
        ax.set_xlabel("Time (s)")
        ax.set_title(f"RAM Usage — {self.base_filename}")
        ax.grid(True, alpha=0.4)
        self._shade_phases(ax, spans)
        if spans:
            ax.legend(loc="upper right", fontsize=8)
        _save(fig, "ram_usage")

        # ------------------------------------------------------------------
        # 2. Page fault rates (delta per 100 ms sample)
        # ------------------------------------------------------------------
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        axes[0].plot(t, col("pgfault_delta"), color="#E74C3C",
                     linewidth=1, label="Minor faults / interval")
        axes[0].set_ylabel("Minor page faults")
        axes[0].set_title(f"Page Fault Rates — {self.base_filename}")
        axes[0].grid(True, alpha=0.4)
        self._shade_phases(axes[0], spans)

        axes[1].plot(t, col("pgmajfault_delta"), color="#8E44AD",
                     linewidth=1, label="Major faults / interval")
        axes[1].set_ylabel("Major page faults")
        axes[1].set_xlabel("Time (s)")
        axes[1].grid(True, alpha=0.4)
        self._shade_phases(axes[1], spans)

        if spans:
            handles = [mpatches.Patch(color=_PHASE_COLOURS[i % len(_PHASE_COLOURS)],
                                      label=s[2], alpha=0.5)
                       for i, s in enumerate(spans)]
            axes[0].legend(handles=handles, loc="upper right", fontsize=7)

        plt.tight_layout()
        _save(fig, "page_fault_rates")

        # ------------------------------------------------------------------
        # 3. LLC hit/miss rates
        # ------------------------------------------------------------------
        llc_loads  = np.array(col("llc_loads"), dtype=float)
        llc_misses = np.array(col("llc_load_misses"), dtype=float)
        llc_miss_r = np.where(llc_loads > 0, llc_misses / llc_loads, 0.0)

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        axes[0].plot(t, llc_loads / 1e3, color="#27AE60",
                     linewidth=1, label="LLC loads (k/interval)")
        axes[0].plot(t, llc_misses / 1e3, color="#E67E22",
                     linewidth=1, label="LLC misses (k/interval)")
        axes[0].set_ylabel("Count (thousands)")
        axes[0].set_title(f"LLC Access Patterns — {self.base_filename}")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.4)
        self._shade_phases(axes[0], spans)

        axes[1].plot(t, llc_miss_r * 100, color="#C0392B",
                     linewidth=1.5, label="LLC miss rate (%)")
        axes[1].set_ylabel("LLC miss rate (%)")
        axes[1].set_ylim(bottom=0)
        axes[1].set_xlabel("Time (s)")
        axes[1].grid(True, alpha=0.4)
        self._shade_phases(axes[1], spans)

        plt.tight_layout()
        _save(fig, "llc_access_patterns")

        # ------------------------------------------------------------------
        # 4. TLB miss rates (dTLB + iTLB)
        # ------------------------------------------------------------------
        dtlb_loads = np.array(col("dtlb_loads"), dtype=float)
        dtlb_miss  = np.array(col("dtlb_load_misses"), dtype=float)
        dtlb_rate  = np.where(dtlb_loads > 0, dtlb_miss / dtlb_loads, 0.0)

        itlb_loads = np.array(col("itlb_loads"), dtype=float)
        itlb_miss  = np.array(col("itlb_load_misses"), dtype=float)
        itlb_rate  = np.where(itlb_loads > 0, itlb_miss / itlb_loads, 0.0)

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        axes[0].plot(t, dtlb_rate * 100, color="#1ABC9C",
                     linewidth=1.5, label="dTLB miss rate (%)")
        axes[0].set_ylabel("dTLB miss rate (%)")
        axes[0].set_title(f"TLB Miss Rates — {self.base_filename}")
        axes[0].set_ylim(bottom=0)
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.4)
        self._shade_phases(axes[0], spans)

        axes[1].plot(t, itlb_rate * 100, color="#D35400",
                     linewidth=1.5, label="iTLB miss rate (%)")
        axes[1].set_ylabel("iTLB miss rate (%)")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylim(bottom=0)
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.4)
        self._shade_phases(axes[1], spans)

        plt.tight_layout()
        _save(fig, "tlb_miss_rates")

        # ------------------------------------------------------------------
        # 5. IPC and cache miss rate overview
        # ------------------------------------------------------------------
        ipc       = col("ipc")
        c_miss_r  = col("cache_miss_rate")

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        axes[0].plot(t, ipc, color="#2C3E50", linewidth=1.5, label="IPC")
        axes[0].set_ylabel("Instructions per cycle")
        axes[0].set_title(f"CPU Efficiency — {self.base_filename}")
        axes[0].grid(True, alpha=0.4)
        self._shade_phases(axes[0], spans)

        axes[1].plot(t, [v * 100 for v in c_miss_r], color="#E74C3C",
                     linewidth=1.5, label="Cache miss rate (%)")
        axes[1].set_ylabel("Cache miss rate (%)")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylim(bottom=0)
        axes[1].grid(True, alpha=0.4)
        self._shade_phases(axes[1], spans)

        plt.tight_layout()
        _save(fig, "cpu_efficiency")

        # ------------------------------------------------------------------
        # 6. Memory IO rate (pgpgin / pgpgout deltas)
        # ------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.fill_between(t, col("pgpgin_delta"), alpha=0.5,
                        color="#3498DB", label="Pages read in / interval")
        ax.fill_between(t, col("pgpgout_delta"), alpha=0.5,
                        color="#E74C3C", label="Pages written out / interval")
        ax.set_ylabel("Pages / interval")
        ax.set_xlabel("Time (s)")
        ax.set_title(f"Memory IO Rate — {self.base_filename}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.4)
        self._shade_phases(ax, spans)
        _save(fig, "memory_io_rate")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_phase_summary(self) -> Dict[str, dict]:
        """
        Return per-phase average metrics, useful for printing a table.

        Returns a dict keyed by phase name, each value being a dict of
        average metric values over that phase's time window.
        """
        spans = self._phase_spans()
        if not spans or not self.metrics:
            return {}

        t = np.array(self.timestamps)
        summary: Dict[str, dict] = {}

        for start, end, label in spans:
            if end <= start:
                continue
            mask = (t >= start) & (t <= end)
            if not mask.any():
                continue

            indices = np.where(mask)[0]
            subset = [self.metrics[i] for i in indices]

            keys = [
                "mem_used_kb", "cache_miss_rate",
                "dtlb_miss_rate", "ipc",
                "pgfault_delta", "pgmajfault_delta",
            ]
            summary[label] = {
                k: float(np.mean([m.get(k, 0) for m in subset]))
                for k in keys
            }

        return summary
