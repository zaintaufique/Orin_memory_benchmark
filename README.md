# Jetson Orin NX Inference Benchmark Suite

This project benchmarks PyTorch vision models on NVIDIA Jetson platforms with an emphasis on:

- Measuring end-to-end inference latency and throughput
- Comparing CPU and GPU execution under fixed clock settings
- Capturing RAM, cache, TLB, page-fault, GPU load, and power behavior during runs
- Profiling per-layer memory access patterns to understand where memory pressure appears

The code is built around Jetson-style Linux `sysfs` and `/proc` interfaces, so it is intended for Jetson Orin-class devices rather than generic desktops.

## Objectives

The repository is designed to answer a few practical questions:

- How fast does a model run on CPU vs GPU?
- How does frequency scaling affect latency, throughput, and energy?
- What happens to RAM usage, cache behavior, and page faults during inference?
- Which model layers are responsible for the biggest memory changes?

In short, this is not only a benchmark runner. It is also a lightweight profiling toolkit for performance and memory-access analysis.

## Main Scripts

- `benchmark.py`
  Runs the full benchmark workflow for either CPU or GPU. It configures frequencies, loads the model, executes inference, collects metrics, and saves CSV/plot outputs.

- `model_runner.py`
  Provides the shared model-loading and inference logic, including image preprocessing, latency plotting, RAM monitoring, GPU monitoring, and per-layer memory profiling.

- `model_configs.py`
  Defines the supported torchvision models and their expected input sizes.

- `cpu_setup_verify.py`
  Checks current CPU governor/frequency settings and optionally switches the CPU into performance mode before benchmarking.

- `cpu_freq_manager.py`
  Reads and controls CPU governors and frequencies through `sysfs`.

- `gpu_freq_manager.py`
  Reads and sets the Jetson GPU maximum frequency through `sysfs`.

- `cpu_monitor.py`
  Records CPU utilization, memory usage, temperature, load average, and CPU frequency during CPU benchmarks.

- `gpu_monitor.py`
  Records CUDA memory usage and GPU load during GPU benchmarks.

- `gpu_power_monitor.py`
  Reads INA3221 sensor channels to estimate per-channel and total power draw.

- `ram_monitor.py`
  Samples RAM usage, `/proc/vmstat`, and `perf stat` counters over time. It also supports phase markers for warmup and inference phases.

- `memory_access_profiler.py`
  Attaches PyTorch forward hooks to modules and records per-layer memory snapshots before and after each layer.

## Supported Models

Models are defined in `model_configs.py` and currently include:

- `vgg-13`
- `vgg-16`
- `vgg-19`
- `efficientnet-b0` through `efficientnet-b7`
- `resnet50`
- `resnet152`
- `inception-v3`

At the moment, `benchmark.py` is hardcoded to test:

```python
models_to_test = [
    'efficientnet-b5'
]
```

If you want to benchmark more models, edit that list in `benchmark.py`.

## Benchmark Modes

There are a few different "modes" in this codebase.

### 1. Device mode

The main CLI mode is chosen with:

```bash
python3 benchmark.py --device cpu
python3 benchmark.py --device gpu
```

Behavior differs by device:

- `cpu`
  - Uses `CPUFrequencyManager`
  - Starts `CPUMonitor`
  - Uses 10 inference iterations by default
  - Tests CPU frequency `1971200` Hz by default

- `gpu`
  - Uses `GPUFrequencyManager`
  - Starts `GPUMonitor` and `GPUPowerMonitor`
  - Uses 100 inference iterations by default
  - Sweeps GPU frequencies `306000000`, `510000000`, and `765000000` Hz

### 2. Frequency-control mode

The benchmark tries to pin the target device to a specific frequency before each run:

- CPU mode sets the governor to `performance` and pins min/max CPU frequency
- GPU mode writes a new GPU max frequency through Jetson GPU `sysfs`

If permissions are missing, the run still proceeds, but frequency control may be skipped and the results may be less controlled.

### 3. Input mode

The runner supports two ways of generating input:

- Real images from a directory
- Random synthetic tensors when no image directory is supplied

In the current `benchmark.py`, `image_dir=None` is passed, so the benchmark uses randomly generated input tensors.

### 4. Precision mode

The unified benchmark in `benchmark.py` forces:

```python
precision = 'fp32'
```

So the current benchmark compares CPU and GPU using FP32 only.

The underlying `ModelRunner` still contains logic for `fp16` image/model handling, but the top-level benchmark path does not use it right now.

### 5. Monitoring / profiling mode

During a run, several layers of monitoring can be active:

- Runtime monitoring
  - CPU metrics or GPU metrics
  - RAM metrics
  - GPU power metrics

- Structural profiling
  - Per-layer memory snapshots from `MemoryAccessProfiler`

- Phase-aware profiling
  - Warmup and inference sections are marked so plots can show where behavior changes

## Requirements

Typical Python dependencies used by these scripts:

```bash
pip install torch torchvision pandas numpy matplotlib pillow psutil
```

System-level requirements on Jetson:

- Linux environment with access to `/proc` and `/sys`
- NVIDIA Jetson GPU and CUDA-enabled PyTorch for GPU mode
- Permission to read power sensors if you want power logging
- Permission to write CPU/GPU frequency nodes if you want clock control
- `perf` installed if you want cache/TLB/cycle/instruction counters from `ram_monitor.py`

Some actions may require `sudo`.

## Quick Start

### Check CPU setup

```bash
python3 cpu_setup_verify.py --check-only
```

### Put CPU into performance mode

```bash
sudo python3 cpu_setup_verify.py --set-performance
```

### Run a CPU benchmark

```bash
python3 benchmark.py --device cpu
```

### Run a GPU benchmark

```bash
python3 benchmark.py --device gpu
```

## Example Workflow

For a more controlled CPU benchmark:

```bash
sudo python3 cpu_setup_verify.py --set-performance --freq 1971200
python3 benchmark.py --device cpu
```

For a GPU sweep:

```bash
python3 benchmark.py --device gpu
```

This will run the selected model at multiple GPU frequencies and generate result CSVs plus plots.

## What the Benchmark Does

At a high level, `benchmark.py` performs the following steps:

1. Parse `--device`
2. Pick device-specific frequency settings and iteration counts
3. Load the model from `MODEL_CONFIGS`
4. Configure CPU or GPU frequency
5. Start monitoring threads
6. Run warmup iterations
7. Run timed inference iterations
8. Save performance, power, device, and RAM metrics
9. Create plots and a summary CSV
10. Restore original frequency settings where possible

This block from `benchmark.py` is the key device-selection logic:

```python
if device == 'cpu':
    frequencies = [1971200]  # 1.97 GHz
    iterations_per_test = 10
    freq_manager = CPUFrequencyManager()
else:
    frequencies = [306000000, 510000000, 765000000]  # 306, 510, 765 MHz
    iterations_per_test = 100
    freq_manager = GPUFrequencyManager()
```

## Code Snippets Explained

### Model configuration

`model_configs.py` stores a simple registry:

```python
'efficientnet-b5': {
    'factory': lambda: models.efficientnet_b5(
        weights=models.EfficientNet_B5_Weights.DEFAULT
    ),
    'size': 456,
    'source': 'torchvision'
}
```

Meaning:

- `factory` builds the pretrained model
- `size` defines the expected image input resolution
- `source` documents where the model comes from

### Random vs real image input

In `ModelRunner.load_images()`:

```python
if not image_dir:
    return torch.randn(batch_size, 3, input_size, input_size, dtype=dtype)
```

This means the benchmark can run without a dataset folder. That is convenient for repeatable synthetic performance tests, but it does not represent end-to-end application preprocessing costs.

### Warmup behavior

Both CPU and GPU paths perform warmup before timed measurement:

```python
for _ in range(10):
    output = model(input_data)
```

This helps reduce startup noise from one-time initialization effects.

### Selected per-layer profiling passes

`model_runner.py` does not profile every iteration at layer granularity. Instead, it samples a subset:

```python
profile_iters = {0, num_iterations // 4,
                 num_iterations // 2, num_iterations - 1}
```

That keeps profiling useful without making the run excessively heavy.

### Power to energy conversion

`benchmark.py` derives energy per inference from average power and latency:

```python
energy_mj = (avg_power_mw * avg_latency_ms) / 1000
```

This is used to report average energy per inference and per image.

## Output Files

The scripts create a `logs/` directory and save per-run artifacts such as:

- `*_performance.csv`
  Summary latency and throughput statistics

- `*_latencies.csv`
  Per-iteration latency samples

- `*_cpu_metrics.csv`
  CPU utilization, frequencies, temperature, and memory usage

- `*_gpu_metrics.csv`
  GPU memory and load samples

- `*_power_metrics.csv`
  INA3221 channel current, voltage, and power readings

- `*_ram_metrics.csv`
  RAM, vmstat, and perf-counter samples

- `*_phase_markers.csv`
  Warmup/inference markers used for annotated plots

- `*_complete_metrics.csv`
  Consolidated run-level metrics, including energy estimates

- `logs/plots/*.png`
  Plots for latency, CPU/GPU activity, power, RAM usage, page faults, cache behavior, and more

- `*_benchmark_results_<timestamp>.csv`
  Top-level summary file written by `benchmark.py`

## Interpreting the Results

The most important columns and outputs are:

- `avg_latency_ms`
  Average time for one inference

- `throughput_inferences_per_sec`
  How many inferences complete per second

- `throughput_images_per_sec`
  Throughput scaled by batch size

- `total_avg_power`
  Mean measured power across all INA3221 channels

- `*_avg_energy_per_inference_mj`
  Estimated energy consumed per inference

- `avg_gpu_load_percent` or `avg_cpu_percent`
  Device utilization during the run

- RAM / cache / TLB plots
  Useful for spotting stalls, memory pressure, miss-rate spikes, or warmup effects

## Help and Troubleshooting

### GPU requested but CUDA is unavailable

If `benchmark.py --device gpu` fails with a CUDA warning:

- Confirm that Jetson CUDA is installed correctly
- Confirm that your PyTorch build has CUDA enabled
- Check `torch.cuda.is_available()` from Python

### Frequency control is ignored

If you see messages about missing permission:

- Run with `sudo` when appropriate
- Confirm the Jetson `sysfs` paths match your device
- Check whether the board exposes the same CPU/GPU frequency nodes expected by these scripts

### No power data is collected

Possible reasons:

- INA3221 paths differ on your board
- Current user cannot read the sensor nodes
- The script disables power monitoring after a permission or path failure

### RAM perf counters are all zero

`ram_monitor.py` relies on `perf stat`. If the counters are missing:

- Make sure `perf` is installed
- Check permissions for performance counters on the device
- Verify the listed events are supported by your kernel/CPU

### Very small test coverage by default

The current benchmark is intentionally narrow:

- Only `efficientnet-b5` is enabled in `benchmark.py`
- CPU mode runs only 10 iterations

Edit `models_to_test`, frequency lists, and iteration counts if you want a broader study.

## Suggested Improvements

If you extend this project, strong next steps would be:

- Move model lists, iterations, frequencies, batch size, and precision into CLI arguments
- Add an `--image-dir` CLI option so real datasets can be used without editing code
- Save environment metadata such as JetPack version, kernel version, and PyTorch build
- Add CSV/JSON summary export for cross-run comparison
- Make the monitor paths configurable for different Jetson variants

## Summary

This codebase gives you a practical benchmarking harness for Jetson inference experiments:

- controlled CPU/GPU frequency testing
- latency and throughput measurement
- RAM/cache/TLB/page-fault observation
- GPU power tracking
- per-layer memory profiling

Use `cpu_setup_verify.py` to prepare the CPU path, then run `benchmark.py --device cpu` or `benchmark.py --device gpu` depending on the experiment you want to perform.
