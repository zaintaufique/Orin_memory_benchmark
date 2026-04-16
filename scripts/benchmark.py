#!/usr/bin/env python3
"""
Unified benchmarking script for neural network models on Jetson platforms
Supports both CPU and GPU benchmarking with comprehensive monitoring
Uses FP32 precision for both CPU and GPU
"""

import sys
import csv
import time
import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from model_runner import ModelRunner

# Import monitoring and management modules
from cpu_freq_manager import CPUFrequencyManager
from cpu_monitor import CPUMonitor
from gpu_monitor import GPUMonitor
from gpu_power_monitor import GPUPowerMonitor
from gpu_freq_manager import GPUFrequencyManager
from ram_monitor import RAMMonitor


def parse_power_metrics(power_csv_path):
    """Parse power metrics CSV and return average power in mW for all channels"""
    try:
        df = pd.read_csv(power_csv_path)
        if len(df) > 0:
            # Skip first few seconds to avoid startup transients
            startup_skip = 3.0  # seconds
            df_stable = df[df['time_seconds'] > startup_skip]
            if len(df_stable) > 10:  # Need enough samples
                df_use = df_stable
            else:
                df_use = df

            # Calculate stats for all channels and total
            power_stats = {}

            # Individual channels
            for ch in [1, 2, 3]:
                if f'power{ch}_mw' in df_use.columns:
                    power_stats[f'ch{ch}_avg_power'] = df_use[f'power{ch}_mw'].mean()
                    power_stats[f'ch{ch}_max_power'] = df_use[f'power{ch}_mw'].max()
                    power_stats[f'ch{ch}_min_power'] = df_use[f'power{ch}_mw'].min()
                else:
                    power_stats[f'ch{ch}_avg_power'] = 0.0
                    power_stats[f'ch{ch}_max_power'] = 0.0
                    power_stats[f'ch{ch}_min_power'] = 0.0

            # Total power
            if 'total_power_mw' in df_use.columns:
                power_stats['total_avg_power'] = df_use['total_power_mw'].mean()
                power_stats['total_max_power'] = df_use['total_power_mw'].max()
                power_stats['total_min_power'] = df_use['total_power_mw'].min()
            else:
                # Calculate total if not present
                total_power = (
                    power_stats['ch1_avg_power'] +
                    power_stats['ch2_avg_power'] +
                    power_stats['ch3_avg_power']
                )
                power_stats['total_avg_power'] = total_power
                power_stats['total_max_power'] = (
                    power_stats['ch1_max_power'] +
                    power_stats['ch2_max_power'] +
                    power_stats['ch3_max_power']
                )
                power_stats['total_min_power'] = (
                    power_stats['ch1_min_power'] +
                    power_stats['ch2_min_power'] +
                    power_stats['ch3_min_power']
                )

            return power_stats
        else:
            print(f"Warning: No power data found in {power_csv_path}")
            default_power = {f'ch{ch}_{stat}_power': 0.0 for ch in [1, 2, 3] for stat in ['avg', 'max', 'min']}
            total_power = {'total_avg_power': 0.0, 'total_max_power': 0.0, 'total_min_power': 0.0}
            return {**default_power, **total_power}
    except Exception as e:
        print(f"Error parsing power metrics from {power_csv_path}: {e}")
        default_power = {f'ch{ch}_{stat}_power': 0.0 for ch in [1, 2, 3] for stat in ['avg', 'max', 'min']}
        total_power = {'total_avg_power': 0.0, 'total_max_power': 0.0, 'total_min_power': 0.0}
        return {**default_power, **total_power}


def parse_performance_metrics(perf_csv_path):
    """Parse performance CSV and return comprehensive metrics"""
    try:
        df = pd.read_csv(perf_csv_path)
        if len(df) > 0:
            row = df.iloc[0]
            return {
                'avg_latency_ms': row.get('avg_latency', 0.0),
                'min_latency_ms': row.get('min_latency', 0.0),
                'max_latency_ms': row.get('max_latency', 0.0),
                'p95_latency_ms': row.get('p95_latency', 0.0),
                'p99_latency_ms': row.get('p99_latency', 0.0),
                'std_latency_ms': row.get('std_latency', 0.0),
                'throughput_inferences_per_sec': row.get('throughput', 0.0),
                'images_per_second': row.get('images_per_second', 0.0),
                'total_time_sec': row.get('total_time', 0.0),
                'num_iterations': row.get('num_iterations', 0),
                'batch_size': row.get('batch_size', 32)
            }
        else:
            return {
                k: 0.0 for k in [
                    'avg_latency_ms', 'min_latency_ms', 'max_latency_ms',
                    'p95_latency_ms', 'p99_latency_ms', 'std_latency_ms',
                    'throughput_inferences_per_sec', 'images_per_second',
                    'total_time_sec', 'num_iterations', 'batch_size'
                ]
            }
    except Exception as e:
        print(f"Error parsing performance metrics: {e}")
        return {
            k: 0.0 for k in [
                'avg_latency_ms', 'min_latency_ms', 'max_latency_ms',
                'p95_latency_ms', 'p99_latency_ms', 'std_latency_ms',
                'throughput_inferences_per_sec', 'images_per_second',
                'total_time_sec', 'num_iterations', 'batch_size'
            ]
        }


def parse_device_metrics(csv_path, device):
    """Unified function to parse device-specific metrics"""
    try:
        df = pd.read_csv(csv_path)
        if len(df) > 0:
            startup_skip = 3.0
            df_stable = df[df['time_seconds'] > startup_skip]
            df_use = df_stable if len(df_stable) > 10 else df

            if device == 'cpu':
                return {
                    'avg_cpu_percent': df_use['cpu_percent_overall'].mean(),
                    'max_cpu_percent': df_use['cpu_percent_overall'].max(),
                    'avg_memory_percent': df_use['memory_percent'].mean(),
                    'max_memory_percent': df_use['memory_percent'].max(),
                    'avg_cpu_temp_celsius': df_use['cpu_temp_celsius'].mean(),
                    'max_cpu_temp_celsius': df_use['cpu_temp_celsius'].max(),
                    'avg_cpu0_freq_mhz': df_use['cpu0_freq_khz'].mean() / 1000,
                    'avg_cpu4_freq_mhz': df_use['cpu4_freq_khz'].mean() / 1000
                }
            else:  # GPU
                return {
                    'avg_gpu_memory_mb': df_use['allocated_mb'].mean(),
                    'max_gpu_memory_mb': df_use['allocated_mb'].max(),
                    'avg_gpu_load_percent': df_use['gpu_load_percent'].mean(),
                    'max_gpu_load_percent': df_use['gpu_load_percent'].max()
                }
        else:
            return {}
    except Exception as e:
        print(f"Error parsing {device} metrics: {e}")
        return {}


class UnifiedModelRunner(ModelRunner):
    """Extended ModelRunner with both CPU and GPU monitoring capabilities"""

    def run_inference(
        self,
        model,
        model_name,
        input_size,
        batch_size=32,
        num_iterations=100,
        device='gpu',
        precision='fp32',
        freq=None,
        image_dir=None
    ):
        """Runs inference with comprehensive monitoring and logging for both CPU and GPU"""
        print(f"\nRunning {device.upper()} inference for {num_iterations} iterations...")
        print(f"Batch size: {batch_size}, Precision: {precision}, Device: {device}")

        # Create frequency-specific filename
        if freq:
            if device.lower() == 'cpu':
                freq_mhz = int(freq / 1000)  # Hz to MHz for CPU
            else:
                freq_mhz = int(freq / 1e6)   # Hz to MHz for GPU
            base_filename = f"{model_name}_{device}_b{batch_size}_{precision}_{freq_mhz}MHz"
        else:
            base_filename = f"{model_name}_{device}_b{batch_size}_{precision}"

        # Start appropriate monitoring with frequency-specific filename
        cpu_monitor = None
        gpu_monitor = None
        ram_monitor = RAMMonitor(self.log_dir, model_name, device, batch_size, precision)
        power_monitor = GPUPowerMonitor(self.log_dir, model_name, device, batch_size, precision)

        # Override monitor base_filename to include frequency
        if freq:
            power_monitor.update_base_filename(base_filename)
            ram_monitor.update_base_filename(base_filename)

        if device.lower() == 'cpu':
            cpu_monitor = CPUMonitor(self.log_dir, model_name, device, batch_size, precision)
            if freq:
                cpu_monitor.update_base_filename(base_filename)
            cpu_monitor.start()
        else:
            gpu_monitor = GPUMonitor(self.log_dir, model_name, device, batch_size, precision)
            if freq:
                gpu_monitor.update_base_filename(base_filename)
            gpu_monitor.start()

        power_monitor.start()
        ram_monitor.start()

        try:
            # Prepare input data
            input_data = self.load_images(image_dir, input_size, batch_size, precision)

            if device.lower() == 'gpu' and torch.cuda.is_available():
                input_data = input_data.cuda()

            latencies = []

            # Warmup runs
            print("Performing warmup runs...")
            with torch.no_grad():
                for _ in range(10):
                    if device.lower() == 'gpu':
                        torch.cuda.synchronize()
                    output = model(input_data)
                    if device.lower() == 'gpu':
                        torch.cuda.synchronize()

            # Main inference loop
            print("Starting performance measurement...")
            start_time = time.time()

            with torch.no_grad():
                for iteration in range(num_iterations):
                    if device.lower() == 'gpu':
                        torch.cuda.synchronize()

                    inference_start = time.time()
                    output = model(input_data)

                    if device.lower() == 'gpu':
                        torch.cuda.synchronize()

                    inference_end = time.time()
                    latency = (inference_end - inference_start) * 1000
                    latencies.append(latency)

                    # Progress update
                    progress_interval = 5 if device.lower() == 'cpu' else 25
                    if (iteration + 1) % progress_interval == 0:
                        print(f"Completed {iteration + 1}/{num_iterations} iterations")
                        print(f"Current avg latency: {np.mean(latencies[-progress_interval:]):.2f} ms")

            end_time = time.time()
            total_time = end_time - start_time

            # Calculate statistics
            avg_latency = np.mean(latencies)
            std_latency = np.std(latencies)
            min_latency = np.min(latencies)
            max_latency = np.max(latencies)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)

            # Print performance summary
            print(f"\n{device.upper()} Inference Performance Summary:")
            print(f"Average latency: {avg_latency:.2f} ms")
            print(f"Throughput: {num_iterations/total_time:.2f} inferences/sec")
            print(f"Images/sec: {(num_iterations * batch_size)/total_time:.2f}")

            # Save performance data with frequency in filename
            perf_path = self.log_dir / f"{base_filename}_performance.csv"

            self.log_dir.mkdir(exist_ok=True)

            with open(perf_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'total_time', 'avg_latency', 'std_latency',
                    'min_latency', 'max_latency', 'p95_latency',
                    'p99_latency', 'throughput', 'images_per_second',
                    'num_iterations', 'batch_size', 'device', 'precision',
                    'frequency_hz', 'frequency_mhz'
                ])
                freq_hz = freq if freq else 0
                freq_mhz_value = freq_mhz if freq else 0
                writer.writerow([
                    total_time, avg_latency, std_latency,
                    min_latency, max_latency, p95_latency,
                    p99_latency, num_iterations / total_time,
                    (num_iterations * batch_size) / total_time,
                    num_iterations, batch_size, device, precision,
                    freq_hz, freq_mhz_value
                ])

            # Save latency data and create plots
            latency_path = self.log_dir / f"{base_filename}_latencies.csv"
            with open(latency_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['iteration', 'latency_ms'])
                for i, latency in enumerate(latencies):
                    writer.writerow([i, latency])

            # Use base_filename for plot (already includes frequency)
            self.plot_latencies(latencies, model_name, device, batch_size, precision, self.log_dir, freq)

            return True

        except Exception as e:
            print(f"Error during inference: {e}")
            return False

        finally:
            if cpu_monitor:
                cpu_monitor.stop()
            if gpu_monitor:
                gpu_monitor.stop()
            if power_monitor:
                power_monitor.stop()
            if ram_monitor:
                ram_monitor.stop()


def save_individual_run_metrics(log_dir, base_filename, power_stats, perf_metrics, device_metrics, freq):
    """Save comprehensive metrics for individual run including energy calculations"""

    # Calculate energy metrics for all channels
    energy_metrics = {}
    for ch in [1, 2, 3]:
        avg_power = power_stats[f'ch{ch}_avg_power']
        min_power = power_stats[f'ch{ch}_min_power']
        max_power = power_stats[f'ch{ch}_max_power']

        # Energy = Power * Time (mW * ms = µJ, convert to mJ)
        energy_metrics[f'ch{ch}_avg_energy_per_inference_mj'] = (avg_power * perf_metrics['avg_latency_ms']) / 1000
        energy_metrics[f'ch{ch}_min_energy_per_inference_mj'] = (min_power * perf_metrics['min_latency_ms']) / 1000
        energy_metrics[f'ch{ch}_max_energy_per_inference_mj'] = (max_power * perf_metrics['max_latency_ms']) / 1000
        energy_metrics[f'ch{ch}_avg_energy_per_image_mj'] = (
            energy_metrics[f'ch{ch}_avg_energy_per_inference_mj'] / perf_metrics['batch_size']
        )

    # Total energy
    total_avg_power = power_stats['total_avg_power']
    total_min_power = power_stats['total_min_power']
    total_max_power = power_stats['total_max_power']

    energy_metrics['total_avg_energy_per_inference_mj'] = (total_avg_power * perf_metrics['avg_latency_ms']) / 1000
    energy_metrics['total_min_energy_per_inference_mj'] = (total_min_power * perf_metrics['min_latency_ms']) / 1000
    energy_metrics['total_max_energy_per_inference_mj'] = (total_max_power * perf_metrics['max_latency_ms']) / 1000
    energy_metrics['total_avg_energy_per_image_mj'] = (
        energy_metrics['total_avg_energy_per_inference_mj'] / perf_metrics['batch_size']
    )

    # Save to CSV
    metrics_path = Path(log_dir) / f"{base_filename}_complete_metrics.csv"
    with open(metrics_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value', 'unit'])

        # Performance metrics
        writer.writerow(['avg_latency', perf_metrics['avg_latency_ms'], 'ms'])
        writer.writerow(['throughput', perf_metrics['throughput_inferences_per_sec'], 'inferences/sec'])
        writer.writerow(['images_per_second', perf_metrics['images_per_second'], 'images/sec'])

        # Power metrics for all channels
        for ch in [1, 2, 3]:
            writer.writerow([f'ch{ch}_avg_power', power_stats[f'ch{ch}_avg_power'], 'mW'])
            writer.writerow([f'ch{ch}_avg_energy_per_inference', energy_metrics[f'ch{ch}_avg_energy_per_inference_mj'], 'mJ'])
            writer.writerow([f'ch{ch}_avg_energy_per_image', energy_metrics[f'ch{ch}_avg_energy_per_image_mj'], 'mJ'])

        # Total metrics
        writer.writerow(['total_avg_power', power_stats['total_avg_power'], 'mW'])
        writer.writerow(['total_avg_energy_per_inference', energy_metrics['total_avg_energy_per_inference_mj'], 'mJ'])
        writer.writerow(['total_avg_energy_per_image', energy_metrics['total_avg_energy_per_image_mj'], 'mJ'])
        writer.writerow(['frequency', (freq / 1e6) if freq else 0, 'MHz'])

    print(f"Complete metrics saved to {metrics_path}")


def run_single_benchmark(runner, freq_manager, model_name, device='gpu', freq=None, batch_size=32, iterations=100):
    """Run a single benchmark and return comprehensive metrics"""

    freq_str = f"{freq/1000:.0f} MHz" if device == 'cpu' else f"{freq/1e6:.0f} MHz"
    print(f"\n  🔄 Running {model_name} on {device.upper()} at {freq_str}...")

    try:
        # Configure frequency
        if device.lower() == 'cpu':
            freq_manager.configure_cpu_performance(freq)
        else:
            freq_manager.set_max_freq(freq)

        # Use FP32 precision for both devices
        precision = 'fp32'
        print(f"  Using {precision.upper()} precision for {device.upper()}")

        # Load model with fp32 precision
        model = runner.load_model(model_name, device=device, precision=precision)
        input_size = runner.model_configs[model_name]['size']

        # Run inference
        success = runner.run_inference(
            model=model,
            model_name=model_name,
            input_size=input_size,
            batch_size=batch_size,
            num_iterations=iterations,
            device=device,
            precision=precision,
            freq=freq,
            image_dir=None
        )

        if not success:
            print(f"    ❌ Inference failed for {model_name}")
            return None

        # Create frequency-specific filename for parsing
        if freq:
            freq_mhz = int(freq / (1000 if device.lower() == 'cpu' else 1e6))
            base_filename = f"{model_name}_{device}_b{batch_size}_{precision}_{freq_mhz}MHz"
        else:
            base_filename = f"{model_name}_{device}_b{batch_size}_{precision}"

        log_dir = runner.log_dir

        power_csv = log_dir / f"{base_filename}_power_metrics.csv"
        perf_csv = log_dir / f"{base_filename}_performance.csv"
        device_csv = log_dir / f"{base_filename}_{device}_metrics.csv"

        time.sleep(2)  # Wait for files

        # Parse all metrics
        power_stats = parse_power_metrics(power_csv) if power_csv.exists() else \
            {**{f'ch{ch}_{stat}_power': 0.0 for ch in [1, 2, 3] for stat in ['avg', 'max', 'min']},
             **{'total_avg_power': 0.0, 'total_max_power': 0.0, 'total_min_power': 0.0}}

        perf_metrics = parse_performance_metrics(perf_csv) if perf_csv.exists() else \
            {k: 0.0 for k in [
                'avg_latency_ms', 'min_latency_ms', 'max_latency_ms',
                'p95_latency_ms', 'p99_latency_ms', 'std_latency_ms',
                'throughput_inferences_per_sec', 'images_per_second',
                'total_time_sec', 'num_iterations', 'batch_size'
            ]}

        device_metrics = parse_device_metrics(device_csv, device) if device_csv.exists() else {}

        # Save comprehensive individual run metrics
        save_individual_run_metrics(log_dir, base_filename, power_stats, perf_metrics, device_metrics, freq)

        # Calculate energy metrics for the summary
        energy_metrics = {}
        for ch in [1, 2, 3]:
            avg_power = power_stats[f'ch{ch}_avg_power']
            min_power = power_stats[f'ch{ch}_min_power']
            max_power = power_stats[f'ch{ch}_max_power']

            energy_metrics[f'ch{ch}_avg_energy_per_inference_mj'] = (avg_power * perf_metrics['avg_latency_ms']) / 1000
            energy_metrics[f'ch{ch}_min_energy_per_inference_mj'] = (min_power * perf_metrics['min_latency_ms']) / 1000
            energy_metrics[f'ch{ch}_max_energy_per_inference_mj'] = (max_power * perf_metrics['max_latency_ms']) / 1000
            energy_metrics[f'ch{ch}_avg_energy_per_image_mj'] = (
                energy_metrics[f'ch{ch}_avg_energy_per_inference_mj'] / batch_size
            )

        total_avg_power = power_stats['total_avg_power']
        print(
            f"    ✅ Power: {total_avg_power:.1f}mW | Latency: {perf_metrics['avg_latency_ms']:.1f}ms | "
            f"Ch1: {energy_metrics['ch1_avg_energy_per_image_mj']:.3f}mJ"
        )

        # Compile results for summary
        result = {
            'model_name': model_name,
            'model_family': model_name.split('-')[0],
            'device': device.lower(),
            'input_size_pixels': input_size,
            'batch_size': batch_size,
            'precision': precision,
            'frequency_hz': freq,
            'frequency_mhz': freq / (1000 if device.lower() == 'cpu' else 1e6),
            'num_iterations': perf_metrics['num_iterations'],
            'test_duration_sec': perf_metrics['total_time_sec'],
            'avg_latency_ms': perf_metrics['avg_latency_ms'],
            'min_latency_ms': perf_metrics['min_latency_ms'],
            'max_latency_ms': perf_metrics['max_latency_ms'],
            'p95_latency_ms': perf_metrics['p95_latency_ms'],
            'p99_latency_ms': perf_metrics['p99_latency_ms'],
            'std_latency_ms': perf_metrics['std_latency_ms'],
            'throughput_inferences_per_sec': perf_metrics['throughput_inferences_per_sec'],
            'throughput_images_per_sec': perf_metrics['images_per_second'],
        }

        result.update(power_stats)
        result.update(energy_metrics)
        result.update(device_metrics)

        return result

    except Exception as e:
        print(f"    ❌ Error running {model_name}: {str(e)}")
        return None


def create_summary_plots(all_results, output_dir):
    """Create summary plots grouped by device"""
    if not all_results:
        return

    plot_dir = Path(output_dir) / "summary_plots"
    plot_dir.mkdir(exist_ok=True)

    # Group by device
    devices = list(set(r['device'] for r in all_results))

    for device in devices:
        device_results = [r for r in all_results if r['device'] == device]
        models = list(set(r['model_name'] for r in device_results))

        # Average latency plot
        avg_latencies = []
        for model in models:
            model_latencies = [r['avg_latency_ms'] for r in device_results if r['model_name'] == model]
            avg_latencies.append(np.mean(model_latencies))

        plt.figure(figsize=(12, 8))
        bars = plt.bar(models, avg_latencies, color='skyblue', edgecolor='navy', alpha=0.7)
        plt.title(f'Average Latency by Model - {device.upper()}', fontsize=16, fontweight='bold')
        plt.xlabel('Model')
        plt.ylabel('Average Latency (ms)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)

        for bar, latency in zip(bars, avg_latencies):
            plt.text(
                bar.get_x() + bar.get_width() / 2.,
                bar.get_height() + max(avg_latencies) * 0.01,
                f'{latency:.2f}',
                ha='center',
                va='bottom',
                fontweight='bold'
            )

        plt.tight_layout()
        plt.savefig(plot_dir / f"average_latency_by_model_{device}.png", dpi=300, bbox_inches='tight')
        plt.close()

    print(f"📊 Summary plots saved to {plot_dir}")


def main():
    parser = argparse.ArgumentParser(description='Unified Neural Network Benchmarking Tool')
    parser.add_argument('--device', choices=['cpu', 'gpu'], required=True, help='Target device for benchmarking')

    args = parser.parse_args()
    device = args.device.lower()

    # Validate device availability
    if device == 'gpu' and not torch.cuda.is_available():
        print("❌ GPU requested but CUDA not available!")
        return 1

    # Test configuration
    models_to_test = [
        'efficientnet-b5'
    ]

    # Device-specific configuration
    if device == 'cpu':
        frequencies = [1971200]  # 1.97 GHz
        iterations_per_test = 10
        freq_manager = CPUFrequencyManager()
    else:
        frequencies = [306000000, 510000000, 765000000]  # 306, 510, 765 MHz
        iterations_per_test = 100
        freq_manager = GPUFrequencyManager()

    batch_size = 32
    precision = 'fp32'  # Use FP32 for both CPU and GPU

    # Display configuration
    print(f"🚀 {device.upper()} Neural Network Benchmarking")
    print("=" * 60)
    print(f"📋 Models: {len(models_to_test)}")
    print(f"⚡ Frequencies: {len(frequencies)}")
    print(f"🔄 Iterations: {iterations_per_test}")
    print(f"🎯 Precision: {precision}")
    print(f"📊 Total tests: {len(models_to_test) * len(frequencies)}")

    try:
        runner = UnifiedModelRunner()
        all_results = []
        total_tests = len(models_to_test) * len(frequencies)
        current_test = 0
        start_time = datetime.now()

        print(f"\n🕐 Starting at {start_time.strftime('%H:%M:%S')}")

        # Run benchmarks
        for model_name in models_to_test:
            print(f"\n📱 {model_name.upper()}")
            print("-" * 40)

            for freq in frequencies:
                current_test += 1
                print(f"[{current_test}/{total_tests}]", end=" ")

                result = run_single_benchmark(
                    runner=runner,
                    freq_manager=freq_manager,
                    model_name=model_name,
                    device=device,
                    freq=freq,
                    batch_size=batch_size,
                    iterations=iterations_per_test
                )

                if result:
                    result['test_number'] = current_test
                    result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    all_results.append(result)

                time.sleep(2)

        # Save results
        if all_results:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = Path(f'{device}_benchmark_results_{timestamp}.csv')

            # CSV headers
            fieldnames = [
                'test_number', 'timestamp', 'model_name', 'model_family', 'device',
                'batch_size', 'precision', 'input_size_pixels', 'num_iterations',
                'frequency_hz', 'frequency_mhz', 'test_duration_sec',
                'ch1_avg_power', 'ch1_min_power', 'ch1_max_power',
                'ch2_avg_power', 'ch2_min_power', 'ch2_max_power',
                'ch3_avg_power', 'ch3_min_power', 'ch3_max_power',
                'total_avg_power', 'total_min_power', 'total_max_power',
                'avg_latency_ms', 'min_latency_ms', 'max_latency_ms',
                'p95_latency_ms', 'p99_latency_ms', 'std_latency_ms',
                'throughput_inferences_per_sec', 'throughput_images_per_sec',
                'ch1_avg_energy_per_inference_mj', 'ch1_min_energy_per_inference_mj',
                'ch1_max_energy_per_inference_mj', 'ch1_avg_energy_per_image_mj',
                'ch2_avg_energy_per_inference_mj', 'ch2_min_energy_per_inference_mj',
                'ch2_max_energy_per_inference_mj', 'ch2_avg_energy_per_image_mj',
                'ch3_avg_energy_per_inference_mj', 'ch3_min_energy_per_inference_mj',
                'ch3_max_energy_per_inference_mj', 'ch3_avg_energy_per_image_mj',
            ]

            # Add device-specific fields
            if device == 'cpu':
                fieldnames.extend([
                    'avg_cpu_percent', 'max_cpu_percent', 'avg_memory_percent',
                    'max_memory_percent', 'avg_cpu_temp_celsius', 'max_cpu_temp_celsius',
                    'avg_cpu0_freq_mhz', 'avg_cpu4_freq_mhz'
                ])
            else:
                fieldnames.extend([
                    'avg_gpu_memory_mb', 'max_gpu_memory_mb',
                    'avg_gpu_load_percent', 'max_gpu_load_percent'
                ])

            with open(output_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for result in all_results:
                    filtered_result = {k: v for k, v in result.items() if k in fieldnames}
                    writer.writerow(filtered_result)

            create_summary_plots(all_results, Path.cwd())

            # Summary report
            print(f"\n📊 {device.upper()} BENCHMARK SUMMARY")
            print("=" * 120)
            print(f"{'Model':<18} {'Freq':<8} {'Power':<10} {'Latency':<9} {'Energy':<11} {'Throughput':<11}")
            print(f"{'Name':<18} {'(MHz)':<8} {'(mW)':<10} {'(ms)':<9} {'(mJ/img)':<11} {'(inf/s)':<11}")
            print("-" * 120)

            for result in all_results:
                print(
                    f"{result['model_name']:<18} "
                    f"{result['frequency_mhz']:<8.0f} "
                    f"{result['total_avg_power']:<10.1f} "
                    f"{result['avg_latency_ms']:<9.2f} "
                    f"{result['ch1_avg_energy_per_image_mj']:<11.3f} "
                    f"{result['throughput_inferences_per_sec']:<11.2f}"
                )

            total_time = datetime.now() - start_time
            print("=" * 120)
            print(f"✅ Completed: {len(all_results)}/{total_tests} tests in {total_time}")
            print(f"📄 Results: {output_file}")

            return 0
        else:
            print(f"\n❌ No successful results!")
            return 1

    except KeyboardInterrupt:
        print(f"\n\n⚠️ Interrupted")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return 1

    finally:
        if 'freq_manager' in locals():
            if device == 'cpu':
                freq_manager.restore_original_settings()
            else:
                freq_manager.restore_original_freq()


if __name__ == '__main__':
    sys.exit(main())
