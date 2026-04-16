import os
import time
import csv
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path

from gpu_monitor import GPUMonitor
from gpu_freq_manager import GPUFrequencyManager
from gpu_power_monitor import GPUPowerMonitor
from model_configs import MODEL_CONFIGS
from ram_monitor import RAMMonitor
from memory_access_profiler import MemoryAccessProfiler

class ModelRunner:
    def __init__(self):
        self.gpu_freq_manager = GPUFrequencyManager()
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
        torch.cuda.reset_peak_memory_stats()
        
        # Create absolute paths for directories
        self.base_dir = Path.cwd()
        self.log_dir = self.base_dir / 'logs'
        
        # Create directories with proper permissions
        for directory in [self.log_dir]:
            directory.mkdir(exist_ok=True)
            try:
                directory.chmod(0o777)
            except Exception as e:
                print(f"Warning: Could not set permissions for {directory}: {e}")
        
        # Store model configurations
        self.model_configs = MODEL_CONFIGS

    def load_images(self, image_dir, input_size, batch_size, precision='fp16'):
        """Load and preprocess images from a directory
        
        Args:
            image_dir (str): Path to directory containing images
            input_size (int): Model input size (height/width)
            batch_size (int): Batch size for inference
            precision (str): Model precision (affects dtype)
            
        Returns:
            torch.Tensor: Preprocessed images ready for inference
        """
        if not image_dir:
            # Generate random data if no image directory is provided
            print("No image directory provided, generating random input data...")
            if precision == 'fp16':
                dtype = torch.float16
            else:
                dtype = torch.float32
            return torch.randn(batch_size, 3, input_size, input_size, dtype=dtype)
        
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
            
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(image_dir.glob(f"*{ext}")))
            
        if not image_files:
            raise ValueError(f"No image files found in {image_dir}")
            
        print(f"Found {len(image_files)} images in {image_dir}")
        
        # Take only the number of images needed for the batch
        image_files = image_files[:batch_size]
        if len(image_files) < batch_size:
            print(f"Warning: Only {len(image_files)} images found, batch will be incomplete")
            
        # Define preprocessing
        preprocess = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Process images
        batch_data = []
        for img_path in image_files:
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
                batch_data.append(img_tensor)
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                
        # Pad batch if needed
        while len(batch_data) < batch_size:
            # Duplicate the last image to fill the batch
            batch_data.append(batch_data[-1] if batch_data else torch.zeros((1, 3, input_size, input_size)))
            
        # Stack tensors into a single batch
        batch_tensor = torch.cat(batch_data, 0)
        
        # Convert to the right precision
        if precision == 'fp16':
            batch_tensor = batch_tensor.half()
            
        return batch_tensor

    def load_model(self, model_name, device='gpu', precision='fp16'):
        """Load and prepare model for inference
        
        Args:
            model_name (str): Name of the model to load
            device (str): Target device ('cpu' or 'gpu')
            precision (str): Precision to use ('fp32' or 'fp16')
            
        Returns:
            torch.nn.Module: Prepared model ready for inference
        """
        if model_name not in self.model_configs:
            raise ValueError(f"Model {model_name} not supported")
        
        config = self.model_configs[model_name]
        
        print(f"Loading {model_name} model from {config['source']} with {precision} precision...")
        try:
            model = config['factory']()
        except Exception as e:
            print(f"Error creating model {model_name}: {e}")
            raise
        
        model.eval()
        
        # Move model to device and set precision
        if device.lower() == 'gpu' and torch.cuda.is_available():
            model = model.cuda()
            if precision == 'fp16':
                model = model.half()
                print("Model loaded on GPU with FP16 precision")
            else:
                print("Model loaded on GPU with FP32 precision")
        else:
            if device.lower() == 'gpu' and not torch.cuda.is_available():
                print("Warning: GPU requested but CUDA not available, using CPU")
            print(f"Model loaded on CPU with {precision.upper()} precision")
        
        return model

    def plot_latencies(self, latencies, model_name, device, batch_size, precision, log_dir, freq=None):
        """Plot latency data and save as image"""
        import matplotlib.pyplot as plt
        import numpy as np
        from pathlib import Path
        
        # Create plot directory
        plot_dir = Path(log_dir) / "plots"
        plot_dir.mkdir(exist_ok=True)
        
        # Prepare the base filename with frequency if provided
        base_filename = f"{model_name}_{device}_b{batch_size}_{precision}"
        if freq:
            if device.lower() == 'gpu':
                freq_mhz = int(freq / 1e6)  # Hz to MHz
            else:  # CPU
                freq_mhz = int(freq / 1000)  # Hz to MHz
            base_filename += f"_{freq_mhz}MHz"
        
        # Create figure with two subplots (line chart and histogram)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Line chart of latencies
        ax1.plot(latencies, 'b-')
        ax1.set_title(f'Inference Latency Over Time - {model_name}')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Latency (ms)')
        ax1.grid(True)
        
        # Add average, p95, and p99 lines
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        ax1.axhline(y=avg_latency, color='g', linestyle='-', label=f'Average: {avg_latency:.2f} ms')
        ax1.axhline(y=p95_latency, color='orange', linestyle='--', label=f'95th: {p95_latency:.2f} ms')
        ax1.axhline(y=p99_latency, color='r', linestyle='--', label=f'99th: {p99_latency:.2f} ms')
        ax1.legend()
        
        # Histogram of latencies
        ax2.hist(latencies, bins=30, alpha=0.7, color='blue')
        ax2.set_title(f'Latency Distribution - {model_name}')
        ax2.set_xlabel('Latency (ms)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True)
        
        # Add vertical lines for statistics
        ax2.axvline(x=avg_latency, color='g', linestyle='-', label=f'Average: {avg_latency:.2f} ms')
        ax2.axvline(x=p95_latency, color='orange', linestyle='--', label=f'95th: {p95_latency:.2f} ms')
        ax2.axvline(x=p99_latency, color='r', linestyle='--', label=f'99th: {p99_latency:.2f} ms')
        ax2.legend()
        
        # Save plot with frequency in filename
        latency_plot_path = plot_dir / f"{base_filename}_latency.png"
        plt.tight_layout()
        plt.savefig(latency_plot_path)
        plt.close()
        
        print(f"Latency plot saved to {latency_plot_path}")

    def run_inference(self, model, model_name, input_size, batch_size=32, num_iterations=100, 
                     device='gpu', precision='fp16', gpu_freq=765000000, image_dir=None):
        """Runs inference with comprehensive error handling and logging"""
        print(f"\nRunning inference for {num_iterations} iterations...")
        print(f"Batch size: {batch_size}")
        print(f"Precision: {precision}")
        print(f"Device: {device}")
        
        # Start monitors
        gpu_monitor = None
        power_monitor = None
        ram_monitor = RAMMonitor(self.log_dir, model_name, device, batch_size, precision)
        ram_monitor.start()

        if device.lower() == 'gpu' and torch.cuda.is_available():
            gpu_monitor = GPUMonitor(self.log_dir, model_name, device, batch_size, precision)
            power_monitor = GPUPowerMonitor(self.log_dir, model_name, device, batch_size, precision)
            gpu_monitor.start()
            power_monitor.start()

            # Set GPU frequency from command-line argument
            self.gpu_freq_manager.set_max_freq(gpu_freq)

        # Attach per-layer memory profiler.
        # max_depth=4 captures block-level granularity without thousands of
        # tiny sub-module events.  Reduce to 2–3 for very deep models.
        base_filename = f"{model_name}_{device}_b{batch_size}_{precision}"
        mem_profiler = MemoryAccessProfiler(
            model, self.log_dir, base_filename, max_depth=4
        )
        mem_profiler.attach()

        try:
            # Prepare input data
            print(f"Preparing input data with shape: [{batch_size}, 3, {input_size}, {input_size}]")

            # Load real images or generate random data
            input_data = self.load_images(image_dir, input_size, batch_size, precision)

            # Move input data to the same device as model
            if device.lower() == 'gpu' and torch.cuda.is_available():
                input_data = input_data.cuda()

            print(f"Input data shape: {input_data.shape}")
            print(f"Input data dtype: {input_data.dtype}")

            # Lists to store timing information
            latencies = []

            # ---- Warmup runs ----
            print("\nPerforming warmup runs...")
            ram_monitor.mark_phase("warmup_start")

            with torch.no_grad():
                for wi in range(10):
                    try:
                        if device.lower() == 'gpu' and torch.cuda.is_available():
                            torch.cuda.synchronize()

                        # Profile only the first warmup pass so we can compare
                        # it against steady-state iterations later.
                        if wi == 0:
                            mem_profiler.start_pass(label="warmup_0")
                        output = model(input_data)
                        if wi == 0:
                            mem_profiler.end_pass()

                        if device.lower() == 'gpu' and torch.cuda.is_available():
                            torch.cuda.synchronize()

                    except Exception as e:
                        print(f"Error during warmup: {e}")
                        return False

            ram_monitor.mark_phase("warmup_end")
            print("Starting performance measurement...")

            # ---- Main inference loop ----
            ram_monitor.mark_phase("inference_start")
            start_time = time.time()

            # Iterations at which we capture a full per-layer profiling pass.
            # We always capture iteration 0 (first real run) and the last one,
            # plus a few in between so the heatmap shows convergence.
            profile_iters = {0, num_iterations // 4,
                             num_iterations // 2, num_iterations - 1}

            with torch.no_grad():
                for iteration in range(num_iterations):
                    try:
                        if device.lower() == 'gpu' and torch.cuda.is_available():
                            torch.cuda.synchronize()

                        inference_start = time.time()

                        # Start a profiling pass at selected iterations
                        profiling_this_iter = iteration in profile_iters
                        if profiling_this_iter:
                            mem_profiler.start_pass(label=f"iter_{iteration}")

                        output = model(input_data)

                        if profiling_this_iter:
                            mem_profiler.end_pass()

                        if device.lower() == 'gpu' and torch.cuda.is_available():
                            torch.cuda.synchronize()

                        inference_end = time.time()

                        # Calculate and store latency
                        latency = (inference_end - inference_start) * 1000
                        latencies.append(latency)

                        # Progress update + RAM phase marker every 25 iters
                        if (iteration + 1) % 25 == 0:
                            print(f"Completed {iteration + 1}/{num_iterations} iterations")
                            current_avg = np.mean(latencies[-25:])
                            print(f"Current running average latency: {current_avg:.2f} ms")
                            ram_monitor.mark_phase(f"iter_{iteration + 1}")

                    except Exception as e:
                        print(f"Error during iteration {iteration}: {e}")
                        return False

            ram_monitor.mark_phase("inference_end")
                        
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
            print("\nInference Performance Summary:")
            print(f"Total inference time: {total_time:.2f} seconds")
            print(f"Average latency: {avg_latency:.2f} ms")
            print(f"Standard deviation: {std_latency:.2f} ms")
            print(f"Minimum latency: {min_latency:.2f} ms")
            print(f"Maximum latency: {max_latency:.2f} ms")
            print(f"95th percentile latency: {p95_latency:.2f} ms")
            print(f"99th percentile latency: {p99_latency:.2f} ms")
            print(f"Throughput: {num_iterations/total_time:.2f} inferences/second")
            print(f"Images per second: {(num_iterations * batch_size)/total_time:.2f} images/second")
            
            # Save performance data
            try:
                # Create filename with model details
                base_filename = f"{model_name}_{device}_b{batch_size}_{precision}"
                perf_path = self.log_dir / f"{base_filename}_performance.csv"
                
                # Ensure log directory exists
                self.log_dir.mkdir(exist_ok=True)
                
                # Save detailed performance data
                with open(perf_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'total_time', 'avg_latency', 'std_latency',
                        'min_latency', 'max_latency', 'p95_latency',
                        'p99_latency', 'throughput', 'images_per_second',
                        'num_iterations', 'batch_size', 'device', 'precision'
                    ])
                    writer.writerow([
                        total_time, avg_latency, std_latency,
                        min_latency, max_latency, p95_latency,
                        p99_latency, num_iterations/total_time,
                        (num_iterations * batch_size)/total_time,
                        num_iterations, batch_size, device, precision
                    ])
                print(f"\nPerformance data saved to {perf_path}")
                
                # Save raw latency data
                latency_path = self.log_dir / f"{base_filename}_latencies.csv"
                with open(latency_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['iteration', 'latency_ms'])
                    for i, latency in enumerate(latencies):
                        writer.writerow([i, latency])
                print(f"Latency data saved to {latency_path}")
                
                # Plot latency data - pass gpu_freq as freq
                self.plot_latencies(latencies, model_name, device, batch_size, precision, self.log_dir, gpu_freq)
                
                return True
                
            except Exception as e:
                print(f"Error saving log files: {e}")
                return False
                
        except Exception as e:
            print(f"Error during inference setup: {e}")
            return False
            
        finally:
            # Detach per-layer hooks before any cleanup
            mem_profiler.detach()

            # Cleanup GPU resources
            if device.lower() == 'gpu' and torch.cuda.is_available():
                self.gpu_freq_manager.restore_original_freq()

            if gpu_monitor is not None:
                gpu_monitor.stop()

            if power_monitor is not None:
                power_monitor.stop()

            # Stop RAM monitor (saves CSV + plots with phase shading)
            ram_monitor.stop()

            # Save per-layer profiling results
            print("\nSaving per-layer memory access profiles...")
            mem_profiler.save_and_plot()

            # Print a quick phase summary table
            phase_summary = ram_monitor.get_phase_summary()
            if phase_summary:
                print("\n── Memory-access phase summary ──────────────────────────────")
                print(f"{'Phase':<22} {'RAM used MB':>11} {'Cache miss%':>11} "
                      f"{'dTLB miss%':>11} {'IPC':>6} {'PgFaults/s':>11}")
                print("─" * 75)
                for phase, stats in phase_summary.items():
                    print(
                        f"{phase:<22} "
                        f"{stats['mem_used_kb']/1024:>11.1f} "
                        f"{stats['cache_miss_rate']*100:>11.2f} "
                        f"{stats['dtlb_miss_rate']*100:>11.2f} "
                        f"{stats['ipc']:>6.2f} "
                        f"{stats['pgfault_delta']*10:>11.0f}"   # ×10 → per-second at 100 ms intervals
                    )
                print("─" * 75)

            # Print top memory-intensive layers
            top_layers = mem_profiler.top_memory_layers(n=5)
            if top_layers:
                print("\n── Top 5 layers by GPU memory delta ─────────────────────────")
                print(f"{'Module':<45} {'Avg delta (MB)':>15} {'Max |delta| (MB)':>16}")
                print("─" * 78)
                for row in top_layers:
                    print(f"{row['module']:<45} {row['avg_delta_mb']:>15.2f} "
                          f"{row['max_delta_mb']:>16.2f}")
                print("─" * 78)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("\nInference run completed")