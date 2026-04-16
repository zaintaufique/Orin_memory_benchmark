import time
import csv
import threading
import torch
import matplotlib.pyplot as plt
from pathlib import Path

class GPUMonitor:
    def __init__(self, log_dir, model_name, device, batch_size, precision):
        self.log_dir = log_dir
        self.base_filename = f"{model_name}_{device}_b{batch_size}_{precision}"
        self.running = False
        self.gpu_metrics = []
        self.timestamps = []
        
    def update_base_filename(self, new_base_filename):
        """Update the base filename for this monitor"""
        self.base_filename = new_base_filename
        
    def _read_gpu_load(self):
        """Read GPU load from sysfs"""
        try:
            with open("/sys/devices/platform/gpu.0/load", 'r') as f:
                # Divide by 10 to get correct percentage (0-100%)
                return int(f.read().strip()) / 10
        except Exception as e:
            print(f"Warning: Could not read GPU load: {e}")
            return 0
    
    def _monitor_thread(self, interval=0.1):
        """Thread function to monitor GPU metrics"""
        start_time = time.time()
        while self.running:
            try:
                # Get memory metrics
                allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
                
                # Get GPU load
                gpu_load = self._read_gpu_load()
                
                # Record metrics and timestamp
                self.gpu_metrics.append((allocated_mb, reserved_mb, gpu_load))
                self.timestamps.append(time.time() - start_time)
                
                time.sleep(interval)
            except Exception as e:
                print(f"Error in monitoring thread: {e}")
                time.sleep(interval)
    
    def start(self):
        """Start GPU monitoring in a separate thread"""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_thread)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("GPU monitoring started")
    
    def stop(self):
        """Stop GPU monitoring and save results"""
        if not self.running:
            return
            
        self.running = False
        if hasattr(self, 'monitor_thread') and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        print("GPU monitoring stopped")
        
        if not self.gpu_metrics:
            print("No GPU metrics collected")
            return
        
        # Ensure log directory exists
        Path(self.log_dir).mkdir(exist_ok=True)
        
        # Save metrics to CSV
        metrics_path = Path(self.log_dir) / f"{self.base_filename}_gpu_metrics.csv"
        with open(metrics_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time_seconds', 'allocated_mb', 'reserved_mb', 'gpu_load_percent'])
            for i, (allocated, reserved, load) in enumerate(self.gpu_metrics):
                writer.writerow([self.timestamps[i], allocated, reserved, load])
        print(f"GPU metrics saved to {metrics_path}")
        
        # Create plots
        self._create_plots()
    
    def _create_plots(self):
        """Create plots of GPU metrics"""
        if not self.gpu_metrics:
            print("No GPU metrics to plot")
            return
        
        # Extract data
        times = self.timestamps
        allocated = [m[0] for m in self.gpu_metrics]
        reserved = [m[1] for m in self.gpu_metrics]
        gpu_load = [m[2] for m in self.gpu_metrics]
        
        # Create plot directory
        plot_dir = Path(self.log_dir) / "plots"
        plot_dir.mkdir(exist_ok=True)
        
        # Memory plot
        plt.figure(figsize=(10, 6))
        plt.plot(times, allocated, label='Allocated Memory (MB)')
        plt.plot(times, reserved, label='Reserved Memory (MB)')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Memory (MB)')
        plt.title(f'GPU Memory Usage - {self.base_filename}')
        plt.legend()
        plt.grid(True)
        memory_plot_path = plot_dir / f"{self.base_filename}_memory.png"
        plt.savefig(memory_plot_path)
        plt.close()
        
        # GPU load plot
        plt.figure(figsize=(10, 6))
        plt.plot(times, gpu_load, label='GPU Load (%)', color='red')
        plt.xlabel('Time (seconds)')
        plt.ylabel('GPU Load (%)')
        plt.ylim(0, 100)  # Set y-axis range from 0 to 100%
        plt.title(f'GPU Load - {self.base_filename}')
        plt.grid(True)
        load_plot_path = plot_dir / f"{self.base_filename}_load.png"
        plt.savefig(load_plot_path)
        plt.close()
        
        print(f"Plots saved to {plot_dir}")