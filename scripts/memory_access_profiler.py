#!/usr/bin/env python3
"""
MemoryAccessProfiler – per-layer memory access profiling for PyTorch models.

Registers forward hooks on every named module so that GPU memory allocation,
reserved memory, and CPU available memory are captured immediately *before*
and *after* each layer's forward pass.  This gives you a layer-by-layer view
of how memory is consumed and released during a single inference run.

Typical usage
-------------
    from memory_access_profiler import MemoryAccessProfiler

    profiler = MemoryAccessProfiler(model, log_dir, base_filename)
    profiler.attach()

    with torch.no_grad():
        # single profiling pass
        profiler.start_pass()
        output = model(input_data)
        profiler.end_pass()

    profiler.detach()
    profiler.save_and_plot()

To profile across multiple iterations and see how patterns stabilise:

    for i in range(num_iterations):
        profiler.start_pass(label=f"iter_{i}")
        output = model(input_data)
        profiler.end_pass()

    profiler.save_and_plot()
"""

import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn


class MemoryAccessProfiler:
    """
    Attaches forward hooks to every named module in *model* and records
    GPU + CPU memory snapshots at each layer boundary.

    Parameters
    ----------
    model        : The nn.Module to profile.
    log_dir      : Directory where CSVs and plots are written.
    base_filename: Prefix for all output files.
    max_depth    : Only hook modules up to this nesting depth (None = all).
                   Keeping depth to 3–4 avoids thousands of sub-module events
                   while still capturing meaningful boundaries.
    skip_leaf    : If True, skip modules with no children (individual Conv /
                   BN layers).  Set to False for maximum granularity.
    """

    def __init__(
        self,
        model: nn.Module,
        log_dir: Union[str, Path],
        base_filename: str,
        max_depth: Optional[int] = 4,
        skip_leaf: bool = False,
    ):
        self.model = model
        self.log_dir = Path(log_dir)
        self.base_filename = base_filename
        self.max_depth = max_depth
        self.skip_leaf = skip_leaf

        self._hooks: List[torch.utils.hooks.RemovableHook] = []

        # List of pass records.  Each pass is a list of layer events.
        # Shape: passes[pass_idx] = list of LayerEvent dicts
        self._passes: List[List[dict]] = []
        self._current_pass: Optional[List[dict]] = None
        self._current_pass_label: str = ""
        self._pass_start_time: float = 0.0

        self._use_cuda = torch.cuda.is_available()

        # Map from hook id → module path string
        self._hook_module_name: Dict[int, str] = {}

    # ------------------------------------------------------------------
    # Hook attachment / detachment
    # ------------------------------------------------------------------

    def attach(self):
        """Register forward hooks on all qualifying modules."""
        if self._hooks:
            return  # already attached

        for name, module in self._named_modules_filtered():
            # Capture 'name' in the closure explicitly
            pre_hook  = self._make_pre_hook(name)
            post_hook = self._make_post_hook(name)

            self._hooks.append(module.register_forward_pre_hook(pre_hook))
            self._hooks.append(module.register_forward_hook(post_hook))

        print(f"MemoryAccessProfiler: attached {len(self._hooks) // 2} "
              f"module hooks (depth≤{self.max_depth})")

    def detach(self):
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        print("MemoryAccessProfiler: hooks detached")

    # ------------------------------------------------------------------
    # Pass management
    # ------------------------------------------------------------------

    def start_pass(self, label: str = ""):
        """Call immediately before model(input) to begin recording a pass."""
        self._current_pass = []
        self._current_pass_label = label or f"pass_{len(self._passes)}"
        self._pass_start_time = time.perf_counter()

    def end_pass(self):
        """Call immediately after model(input) to finalise the pass."""
        if self._current_pass is None:
            return
        for event in self._current_pass:
            event["pass_label"] = self._current_pass_label
        self._passes.append(self._current_pass)
        self._current_pass = None

    # ------------------------------------------------------------------
    # Persistence and visualisation
    # ------------------------------------------------------------------

    def save_and_plot(self):
        """Write CSVs and generate all plots for all recorded passes."""
        if not self._passes:
            print("MemoryAccessProfiler: no passes recorded")
            return

        self.log_dir.mkdir(exist_ok=True)
        self._save_csv()
        self._plot_layer_timeline()
        self._plot_pass_comparison()
        self._plot_layer_delta_heatmap()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _named_modules_filtered(self):
        """Yield (name, module) pairs respecting max_depth and skip_leaf."""
        for name, module in self.model.named_modules():
            if not name:          # skip the root module itself
                continue
            depth = name.count(".") + 1
            if self.max_depth is not None and depth > self.max_depth:
                continue
            has_children = len(list(module.children())) > 0
            if self.skip_leaf and not has_children:
                continue
            yield name, module

    def _snapshot(self) -> dict:
        """Capture an instantaneous memory snapshot."""
        snap: dict = {}

        if self._use_cuda:
            snap["gpu_allocated_mb"] = torch.cuda.memory_allocated() / 1e6
            snap["gpu_reserved_mb"]  = torch.cuda.memory_reserved()  / 1e6
            snap["gpu_peak_mb"]      = torch.cuda.max_memory_allocated() / 1e6
        else:
            snap["gpu_allocated_mb"] = 0.0
            snap["gpu_reserved_mb"]  = 0.0
            snap["gpu_peak_mb"]      = 0.0

        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        snap["cpu_available_mb"] = int(line.split()[1]) / 1024
                        break
        except Exception:
            snap["cpu_available_mb"] = 0.0

        snap["wall_time"] = time.perf_counter()
        return snap

    def _make_pre_hook(self, module_name: str):
        """Return a pre-forward hook that records memory *before* the layer."""
        def hook(module, inputs):
            if self._current_pass is None:
                return
            snap = self._snapshot()
            snap["module"]     = module_name
            snap["event"]      = "pre"
            snap["pass_label"] = self._current_pass_label
            snap["rel_time"]   = snap["wall_time"] - self._pass_start_time
            self._current_pass.append(snap)
        return hook

    def _make_post_hook(self, module_name: str):
        """Return a post-forward hook that records memory *after* the layer."""
        def hook(module, inputs, output):
            if self._current_pass is None:
                return
            snap = self._snapshot()
            snap["module"]     = module_name
            snap["event"]      = "post"
            snap["pass_label"] = self._current_pass_label
            snap["rel_time"]   = snap["wall_time"] - self._pass_start_time
            self._current_pass.append(snap)
        return hook

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    def _save_csv(self):
        all_events = [e for p in self._passes for e in p]
        if not all_events:
            return

        csv_path = self.log_dir / f"{self.base_filename}_layer_memory.csv"
        fieldnames = [
            "pass_label", "module", "event", "rel_time",
            "gpu_allocated_mb", "gpu_reserved_mb", "gpu_peak_mb",
            "cpu_available_mb",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_events)

        print(f"Layer memory events saved to {csv_path}")

    # ------------------------------------------------------------------
    # Plot 1 – single-pass layer timeline
    # ------------------------------------------------------------------

    def _plot_layer_timeline(self):
        """
        For the *last* recorded pass, draw GPU-allocated MB vs. elapsed time,
        annotating key layer boundaries.
        """
        pass_data = self._passes[-1]
        if not pass_data:
            return

        post_events = [e for e in pass_data if e["event"] == "post"]
        if not post_events:
            return

        times   = [e["rel_time"]           for e in post_events]
        gpu_alloc = [e["gpu_allocated_mb"] for e in post_events]
        modules = [e["module"]             for e in post_events]

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(times, gpu_alloc, color="#2980B9", linewidth=1.5,
                label="GPU allocated (MB)")
        ax.fill_between(times, gpu_alloc, alpha=0.15, color="#2980B9")

        # Annotate every ~10th module to avoid clutter
        step = max(1, len(post_events) // 15)
        for i in range(0, len(post_events), step):
            ax.annotate(
                modules[i].split(".")[-1],          # last component only
                xy=(times[i], gpu_alloc[i]),
                xytext=(0, 8), textcoords="offset points",
                fontsize=6, ha="center", color="#555",
                arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.5),
            )

        ax.set_xlabel("Time within forward pass (s)")
        ax.set_ylabel("GPU allocated (MB)")
        ax.set_title(f"Layer-by-layer GPU Memory — "
                     f"{self.base_filename} [{post_events[0]['pass_label']}]")
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=9)

        plot_dir = self.log_dir / "plots"
        plot_dir.mkdir(exist_ok=True)
        path = plot_dir / f"{self.base_filename}_layer_timeline.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path.name}")

    # ------------------------------------------------------------------
    # Plot 2 – per-pass GPU memory comparison
    # ------------------------------------------------------------------

    def _plot_pass_comparison(self):
        """
        Overlay the GPU-allocated-MB curve for several passes on one chart so
        you can see how the memory access pattern stabilises after warmup.
        """
        if len(self._passes) < 2:
            return

        # Show first 3 and last pass at most to keep the chart readable
        indices = sorted(set([0, 1, 2, len(self._passes) - 1]))
        indices = [i for i in indices if i < len(self._passes)]

        colours = ["#E74C3C", "#F39C12", "#27AE60", "#2980B9"]

        fig, ax = plt.subplots(figsize=(14, 5))
        for ci, pi in enumerate(indices):
            pass_data  = self._passes[pi]
            post_events = [e for e in pass_data if e["event"] == "post"]
            if not post_events:
                continue
            label  = post_events[0]["pass_label"]
            times  = [e["rel_time"]           for e in post_events]
            alloc  = [e["gpu_allocated_mb"]   for e in post_events]
            ax.plot(times, alloc, linewidth=1.5,
                    color=colours[ci % len(colours)], label=label)

        ax.set_xlabel("Time within forward pass (s)")
        ax.set_ylabel("GPU allocated (MB)")
        ax.set_title(f"GPU Memory Pattern Across Passes — {self.base_filename}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.35)

        plot_dir = self.log_dir / "plots"
        plot_dir.mkdir(exist_ok=True)
        path = plot_dir / f"{self.base_filename}_pass_comparison.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path.name}")

    # ------------------------------------------------------------------
    # Plot 3 – per-layer GPU memory delta heatmap
    # ------------------------------------------------------------------

    def _plot_layer_delta_heatmap(self):
        """
        For each recorded pass, compute the GPU-memory *delta* (MB) at each
        post-layer event and render as a heatmap (layers × passes).

        Positive delta = memory was allocated by that layer.
        Negative delta = memory was freed.
        """
        # Collect unique module names in execution order from first pass
        first_post = [e for e in self._passes[0] if e["event"] == "post"]
        module_order = [e["module"] for e in first_post]

        if not module_order:
            return

        # Build matrix: rows = modules, columns = passes
        n_modules = len(module_order)
        n_passes  = len(self._passes)
        matrix    = np.zeros((n_modules, n_passes), dtype=float)

        for pi, pass_data in enumerate(self._passes):
            post_events = [e for e in pass_data if e["event"] == "post"]
            pre_events  = {e["module"]: e for e in pass_data if e["event"] == "pre"}

            for mi, module_name in enumerate(module_order):
                post = next((e for e in post_events if e["module"] == module_name), None)
                pre  = pre_events.get(module_name)
                if post and pre:
                    matrix[mi, pi] = (post["gpu_allocated_mb"]
                                      - pre["gpu_allocated_mb"])

        # Only plot top-N rows by absolute max delta to keep heatmap readable
        max_per_row = np.abs(matrix).max(axis=1)
        top_n = min(40, n_modules)
        top_indices = np.argsort(max_per_row)[-top_n:][::-1]

        sub_matrix  = matrix[top_indices]
        sub_labels  = [module_order[i].split(".")[-2:]  # last 2 name parts
                       for i in top_indices]
        sub_labels  = [".".join(parts) for parts in sub_labels]

        pass_labels = []
        for pass_data in self._passes:
            first = next((e for e in pass_data if e.get("pass_label")), None)
            pass_labels.append(first["pass_label"] if first else "")

        fig_h = max(6, top_n * 0.35)
        fig, ax = plt.subplots(figsize=(max(8, n_passes * 0.8 + 3), fig_h))

        vmax = np.abs(sub_matrix).max() or 1.0
        img = ax.imshow(sub_matrix, aspect="auto", cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax)
        plt.colorbar(img, ax=ax, label="GPU memory delta (MB)")

        ax.set_yticks(range(top_n))
        ax.set_yticklabels(sub_labels, fontsize=7)
        ax.set_xticks(range(n_passes))
        ax.set_xticklabels(pass_labels, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("Inference pass")
        ax.set_ylabel("Module (top by |delta|)")
        ax.set_title(f"Per-layer GPU Memory Delta — {self.base_filename}\n"
                     "Blue = freed  |  Red = allocated")

        plt.tight_layout()
        plot_dir = self.log_dir / "plots"
        plot_dir.mkdir(exist_ok=True)
        path = plot_dir / f"{self.base_filename}_layer_delta_heatmap.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path.name}")

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------

    def top_memory_layers(self, n: int = 10) -> List[dict]:
        """
        Return the top-N modules by average GPU allocation delta across
        all recorded passes, useful for logging a quick summary table.
        """
        if not self._passes:
            return []

        first_post = [e for e in self._passes[0] if e["event"] == "post"]
        modules = [e["module"] for e in first_post]

        deltas: Dict[str, List[float]] = {m: [] for m in modules}
        for pass_data in self._passes:
            post_map = {e["module"]: e for e in pass_data if e["event"] == "post"}
            pre_map  = {e["module"]: e for e in pass_data if e["event"] == "pre"}
            for m in modules:
                post = post_map.get(m)
                pre  = pre_map.get(m)
                if post and pre:
                    deltas[m].append(
                        post["gpu_allocated_mb"] - pre["gpu_allocated_mb"]
                    )

        ranked = sorted(
            [
                {
                    "module": m,
                    "avg_delta_mb": float(np.mean(v)) if v else 0.0,
                    "max_delta_mb": float(np.max(np.abs(v))) if v else 0.0,
                }
                for m, v in deltas.items()
            ],
            key=lambda x: abs(x["avg_delta_mb"]),
            reverse=True,
        )
        return ranked[:n]
