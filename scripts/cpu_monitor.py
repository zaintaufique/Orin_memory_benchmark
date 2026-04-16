#!/usr/bin/env python3
"""
CPU Monitor for Jetson Orin platforms
Monitors CPU utilization, temperature, and frequency during benchmarking
"""

import time
import csv
import threading
import psutil
import matplotlib.pyplot as plt
from pathlib import Path

class CPUMonitor:
    def __init__(self, log_dir, model_name, device, batch_size, precision):
        self.log_dir = log_dir
        self.base_filename = f"{model_name}_{device}_b{batch_size}_{precision}"
        self.running = False
        self.cpu_metrics = []
        self.timestamps = []
        
    def update_base_filename(self, new_base_filename):
        """Update the base filename for this monitor"""
        self.base_filename = new_base_filename
        
    def _read_cpu_temp(self):
        """Read CPU temperature from thermal sensors"""
        try:
            # Try different thermal sensor paths for Jetson Orin
            temp_paths = [
                "/sys/class/thermal/thermal_zone0/temp",
                "/sys/class/thermal/thermal_zone1/temp",
                "/sys/class/thermal/thermal_zone2/temp"
            ]
            
            temperatures = []
            for path in temp_paths:
                try:
                    with open(path, 'r') as f:
                        temp_millic = int(f.read().strip())
                        temp_celsius = temp_millic / 1000.0
                        temperatures.append(temp_celsius)
                except:
                    continue
                    
            if temperatures:
                return max(temperatures)  # Return highest temperature
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def _read_cpu_freq(self):
        """Read current CPU frequencies"""
        try:
            freqs = {}
            cpu_paths = [
                ("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", "cpu0"),
                ("/sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq", "cpu4")
            ]
            
            for path, cpu_name in cpu_paths:
                try:
                    with open(path, 'r') as f:
                        freq_khz = int(f.read().strip())
                        freqs[cpu_name] = freq_khz
                except:
                    freqs[cpu_name] = 0
                    
            return freqs
        except Exception:
            return {"cpu0": 0, "cpu4": 0}
    
    def _monitor_thread(self, interval=0.1):
        """Thread function to monitor CPU metrics"""
        start_time = time.time()
        
        while self.running:
            try:
                # Get CPU utilization (overall and per-core)
                cpu_percent_overall = psutil.cpu_percent(interval=None)
                cpu_percent_per_core = psutil.cpu_percent(interval=None, percpu=True)
                
                # Get memory usage
                memory = psutil.virtual_memory()
                memory_used_mb = memory.used / (1024 * 1024)
                memory_percent = memory.percent
                
                # Get CPU temperature
                cpu_temp = self._read_cpu_temp()
                
                # Get CPU frequencies
                cpu_freqs = self._read_cpu_freq()
                
                # Get load averages
                load_avg = psutil.getloadavg()
                
                # Record comprehensive metrics
                metrics = {
                    'cpu_percent_overall': cpu_percent_overall,
                    'cpu_percent_core0': cpu_percent_per_core[0] if len(cpu_percent_per_core) > 0 else 0,
                    'cpu_percent_core1': cpu_percent_per_core[1] if len(cpu_percent_per_core) > 1 else 0,
                    'cpu_percent_core4': cpu_percent_per_core[4] if len(cpu_percent_per_core) > 4 else 0,
                    'cpu_percent_core5': cpu_percent_per_core[5] if len(cpu_percent_per_core) > 5 else 0,
                    'memory_used_mb': memory_used_mb,
                    'memory_percent': memory_percent,
                    'cpu_temp_celsius': cpu_temp,
                    'cpu0_freq_khz': cpu_freqs.get('cpu0', 0),
                    'cpu4_freq_khz': cpu_freqs.get('cpu4', 0),
                    'load_avg_1min': load_avg[0],
                    'load_avg_5min': load_avg[1],
                    'load_avg_15min': load_avg[2]
                }
                
                self.cpu_metrics.append(metrics)
                self.timestamps.append(time.time() - start_time)
                
                time.sleep(interval)
                
            except Exception as e:
                print(f"Error in CPU monitoring thread: {e}")
                time.sleep(interval)
    
    def start(self):
        """Start CPU monitoring in a separate thread"""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_thread)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("CPU monitoring started")
    
    def stop(self):
        """Stop CPU monitoring and save results"""
        if not self.running:
            return
            
        self.running = False
        if hasattr(self, 'monitor_thread') and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        print("CPU monitoring stopped")
        
        if not self.cpu_metrics:
            print("No CPU metrics collected")
            return
        
        # Ensure log directory exists
        Path(self.log_dir).mkdir(exist_ok=True)
        
        # Save metrics to CSV
        metrics_path = Path(self.log_dir) / f"{self.base_filename}_cpu_metrics.csv"
        with open(metrics_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            header = ['time_seconds'] + list(self.cpu_metrics[0].keys())
            writer.writerow(header)
            
            # Write data
            for i, metrics in enumerate(self.cpu_metrics):
                row = [self.timestamps[i]] + list(metrics.values())
                writer.writerow(row)
                
        print(f"CPU metrics saved to {metrics_path}")
        
        # Create plots
        self._create_plots()
    
    def _create_plots(self):
        """Create plots of CPU metrics"""
        if not self.cpu_metrics:
            print("No CPU metrics to plot")
            return
        
        # Extract data for plotting
        times = self.timestamps
        cpu_overall = [m['cpu_percent_overall'] for m in self.cpu_metrics]
        cpu_core0 = [m['cpu_percent_core0'] for m in self.cpu_metrics]
        cpu_core4 = [m['cpu_percent_core4'] for m in self.cpu_metrics]
        memory_percent = [m['memory_percent'] for m in self.cpu_metrics]
        cpu_temp = [m['cpu_temp_celsius'] for m in self.cpu_metrics]
        cpu0_freq = [m['cpu0_freq_khz'] / 1000 for m in self.cpu_metrics]  # Convert to MHz
        cpu4_freq = [m['cpu4_freq_khz'] / 1000 for m in self.cpu_metrics]  # Convert to MHz
        
        # Create plot directory
        plot_dir = Path(self.log_dir) / "plots"
        plot_dir.mkdir(exist_ok=True)
        
        # CPU Utilization plot
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(times, cpu_overall, label='Overall CPU %', linewidth=2)
        plt.plot(times, cpu_core0, label='Core 0 %', alpha=0.7)
        plt.plot(times, cpu_core4, label='Core 4 %', alpha=0.7)
        plt.xlabel('Time (seconds)')
        plt.ylabel('CPU Utilization (%)')
        plt.title(f'CPU Utilization - {self.base_filename}')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 100)
        
        # Memory Usage plot
        plt.subplot(2, 2, 2)
        plt.plot(times, memory_percent, label='Memory Usage %', color='red')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Memory Usage (%)')
        plt.title(f'Memory Usage - {self.base_filename}')
        plt.grid(True)
        plt.ylim(0, 100)
        
        # CPU Temperature plot
        plt.subplot(2, 2, 3)
        plt.plot(times, cpu_temp, label='CPU Temperature', color='orange')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Temperature (°C)')
        plt.title(f'CPU Temperature - {self.base_filename}')
        plt.grid(True)
        
        # CPU Frequency plot
        plt.subplot(2, 2, 4)
        plt.plot(times, cpu0_freq, label='CPU0 Freq (MHz)', color='blue')
        plt.plot(times, cpu4_freq, label='CPU4 Freq (MHz)', color='green')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency (MHz)')
        plt.title(f'CPU Frequency - {self.base_filename}')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        cpu_plot_path = plot_dir / f"{self.base_filename}_cpu_metrics.png"
        plt.savefig(cpu_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"CPU metrics plot saved to {cpu_plot_path}")
        
    def get_average_metrics(self):
        """Get average values for key metrics (for summary reporting)"""
        if not self.cpu_metrics:
            return {}
            
        # Skip first few seconds to avoid startup transients
        startup_skip = 3.0
        stable_metrics = []
        for i, timestamp in enumerate(self.timestamps):
            if timestamp > startup_skip:
                stable_metrics.append(self.cpu_metrics[i])
                
        if not stable_metrics:
            stable_metrics = self.cpu_metrics
            
        return {
            'avg_cpu_percent': sum(m['cpu_percent_overall'] for m in stable_metrics) / len(stable_metrics),
            'max_cpu_percent': max(m['cpu_percent_overall'] for m in stable_metrics),
            'avg_memory_percent': sum(m['memory_percent'] for m in stable_metrics) / len(stable_metrics),
            'max_memory_percent': max(m['memory_percent'] for m in stable_metrics),
            'avg_cpu_temp': sum(m['cpu_temp_celsius'] for m in stable_metrics) / len(stable_metrics),
            'max_cpu_temp': max(m['cpu_temp_celsius'] for m in stable_metrics),
            'avg_cpu0_freq_mhz': sum(m['cpu0_freq_khz'] for m in stable_metrics) / len(stable_metrics) / 1000,
            'avg_cpu4_freq_mhz': sum(m['cpu4_freq_khz'] for m in stable_metrics) / len(stable_metrics) / 1000
        }