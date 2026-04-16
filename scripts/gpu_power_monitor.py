import time
import csv
import threading
import matplotlib.pyplot as plt
from pathlib import Path

class GPUPowerMonitor:
    def __init__(self, log_dir, model_name, device, batch_size, precision):
        self.log_dir = log_dir
        self.base_filename = f"{model_name}_{device}_b{batch_size}_{precision}"
        self.running = False
        self.power_metrics = []
        self.timestamps = []
        self.has_permission = True
        
        # INA3221 device paths for all three channels
        self.current1_path = "/sys/bus/i2c/devices/1-0040/hwmon/hwmon4/curr1_input"
        self.voltage1_path = "/sys/bus/i2c/devices/1-0040/hwmon/hwmon4/in1_input"
        self.current2_path = "/sys/bus/i2c/devices/1-0040/hwmon/hwmon4/curr2_input"
        self.voltage2_path = "/sys/bus/i2c/devices/1-0040/hwmon/hwmon4/in2_input"
        self.current3_path = "/sys/bus/i2c/devices/1-0040/hwmon/hwmon4/curr3_input"
        self.voltage3_path = "/sys/bus/i2c/devices/1-0040/hwmon/hwmon4/in3_input"
        
        # Check if we have permission to read the power sensors
        self._check_permissions()
        
    def update_base_filename(self, new_base_filename):
        """Update the base filename for this monitor"""
        self.base_filename = new_base_filename
        
    def _check_permissions(self):
        """Check if we have permissions to read power metrics"""
        try:
            # Test all channels
            for current_path, voltage_path in [
                (self.current1_path, self.voltage1_path),
                (self.current2_path, self.voltage2_path),
                (self.current3_path, self.voltage3_path)
            ]:
                with open(current_path, 'r') as f:
                    _ = f.read().strip()
                with open(voltage_path, 'r') as f:
                    _ = f.read().strip()
            self.has_permission = True
            return True
        except PermissionError:
            print("Note: Running without GPU power monitoring (requires sudo or permission changes)")
            self.has_permission = False
            return False
        except Exception as e:
            print(f"Warning: Power monitoring disabled due to error: {e}")
            self.has_permission = False
            return False
            
    def _read_power(self):
        """Read GPU power (current × voltage) from all INA3221 sensor channels"""
        if not self.has_permission:
            return (0, 0, 0, 0, 0, 0, 0, 0, 0)  # ch1_current, ch1_voltage, ch1_power, ch2_current, ch2_voltage, ch2_power, ch3_current, ch3_voltage, ch3_power
            
        try:
            # Channel 1
            with open(self.current1_path, 'r') as f:
                current1_ma = int(f.read().strip())
            with open(self.voltage1_path, 'r') as f:
                voltage1_mv = int(f.read().strip())
            power1_mw = (current1_ma * voltage1_mv) / 1000
            
            # Channel 2
            with open(self.current2_path, 'r') as f:
                current2_ma = int(f.read().strip())
            with open(self.voltage2_path, 'r') as f:
                voltage2_mv = int(f.read().strip())
            power2_mw = (current2_ma * voltage2_mv) / 1000
            
            # Channel 3
            with open(self.current3_path, 'r') as f:
                current3_ma = int(f.read().strip())
            with open(self.voltage3_path, 'r') as f:
                voltage3_mv = int(f.read().strip())
            power3_mw = (current3_ma * voltage3_mv) / 1000
            
            return (current1_ma, voltage1_mv, power1_mw, 
                   current2_ma, voltage2_mv, power2_mw,
                   current3_ma, voltage3_mv, power3_mw)
            
        except Exception as e:
            # Just return zeros without printing warning every time
            return (0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    def _monitor_thread(self, interval=0.1):
        """Thread function to monitor GPU power metrics"""
        if not self.has_permission:
            # Sleep for a short time to avoid busy waiting
            time.sleep(1)
            return
            
        start_time = time.time()
        while self.running:
            try:
                # Get power metrics for all channels
                (current1_ma, voltage1_mv, power1_mw,
                 current2_ma, voltage2_mv, power2_mw,
                 current3_ma, voltage3_mv, power3_mw) = self._read_power()
                
                # Record metrics and timestamp
                self.power_metrics.append((current1_ma, voltage1_mv, power1_mw,
                                         current2_ma, voltage2_mv, power2_mw,
                                         current3_ma, voltage3_mv, power3_mw))
                self.timestamps.append(time.time() - start_time)
                
                time.sleep(interval)
            except Exception as e:
                # Log error but continue monitoring
                print(f"Error in power monitoring thread: {e}")
                time.sleep(interval)
    
    def start(self):
        """Start GPU power monitoring in a separate thread"""
        if not self.has_permission:
            print("GPU power monitoring disabled (no permission)")
            return
            
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_thread)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("GPU power monitoring started")
    
    def stop(self):
        """Stop GPU power monitoring and save results"""
        if not self.running:
            return
            
        self.running = False
        if hasattr(self, 'monitor_thread') and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        
        if not self.has_permission:
            print("GPU power monitoring was disabled (permission issues)")
            return
            
        print("GPU power monitoring stopped")
        
        if not self.power_metrics:
            print("No GPU power metrics collected")
            return
        
        # Ensure log directory exists
        Path(self.log_dir).mkdir(exist_ok=True)
        
        # Save metrics to CSV
        metrics_path = Path(self.log_dir) / f"{self.base_filename}_power_metrics.csv"
        with open(metrics_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time_seconds', 
                           'current1_ma', 'voltage1_mv', 'power1_mw',
                           'current2_ma', 'voltage2_mv', 'power2_mw', 
                           'current3_ma', 'voltage3_mv', 'power3_mw',
                           'total_power_mw'])
            for i, (current1, voltage1, power1,
                   current2, voltage2, power2,
                   current3, voltage3, power3) in enumerate(self.power_metrics):
                total_power = power1 + power2 + power3
                writer.writerow([self.timestamps[i], 
                               current1, voltage1, power1,
                               current2, voltage2, power2,
                               current3, voltage3, power3,
                               total_power])
        print(f"GPU power metrics saved to {metrics_path}")
        
        # Create plots
        self._create_plots()
    
    def _create_plots(self):
        """Create plots of GPU power metrics"""
        if not self.power_metrics:
            print("No GPU power metrics to plot")
            return
        
        # Extract data
        times = self.timestamps
        power1 = [m[2] for m in self.power_metrics]  # power1_mw
        power2 = [m[5] for m in self.power_metrics]  # power2_mw  
        power3 = [m[8] for m in self.power_metrics]  # power3_mw
        total_power = [p1 + p2 + p3 for p1, p2, p3 in zip(power1, power2, power3)]
        
        current1 = [m[0] for m in self.power_metrics]
        voltage1 = [m[1] for m in self.power_metrics]
        current2 = [m[3] for m in self.power_metrics]
        voltage2 = [m[4] for m in self.power_metrics]
        current3 = [m[6] for m in self.power_metrics]
        voltage3 = [m[7] for m in self.power_metrics]
        
        # Create plot directory
        plot_dir = Path(self.log_dir) / "plots"
        plot_dir.mkdir(exist_ok=True)
        
        # Power plot for all channels
        plt.figure(figsize=(12, 8))
        plt.plot(times, power1, label='Channel 1 Power (mW)', color='blue')
        plt.plot(times, power2, label='Channel 2 Power (mW)', color='red')
        plt.plot(times, power3, label='Channel 3 Power (mW)', color='green')
        plt.plot(times, total_power, label='Total Power (mW)', color='black', linewidth=2)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Power (mW)')
        plt.title(f'GPU Power Consumption - All Channels - {self.base_filename}')
        plt.legend()
        plt.grid(True)
        power_plot_path = plot_dir / f"{self.base_filename}_power_all_channels.png"
        plt.savefig(power_plot_path)
        plt.close()
        
        # Current and Voltage plots for all channels
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Current plot
        ax1.plot(times, current1, 'b-', label='Channel 1 Current (mA)')
        ax1.plot(times, current2, 'r-', label='Channel 2 Current (mA)')
        ax1.plot(times, current3, 'g-', label='Channel 3 Current (mA)')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Current (mA)')
        ax1.set_title(f'Current - All Channels - {self.base_filename}')
        ax1.legend()
        ax1.grid(True)
        
        # Voltage plot
        ax2.plot(times, voltage1, 'b-', label='Channel 1 Voltage (mV)')
        ax2.plot(times, voltage2, 'r-', label='Channel 2 Voltage (mV)')
        ax2.plot(times, voltage3, 'g-', label='Channel 3 Voltage (mV)')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Voltage (mV)')
        ax2.set_title(f'Voltage - All Channels - {self.base_filename}')
        ax2.legend()
        ax2.grid(True)
        
        # Individual channel power plots
        ax3.plot(times, power1, 'b-', label='Channel 1')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Power (mW)')
        ax3.set_title('Channel 1 Power')
        ax3.grid(True)
        
        ax4.plot(times, total_power, 'k-', linewidth=2)
        ax4.set_xlabel('Time (seconds)')
        ax4.set_ylabel('Total Power (mW)')
        ax4.set_title('Total Power (All Channels)')
        ax4.grid(True)
        
        plt.tight_layout()
        cv_plot_path = plot_dir / f"{self.base_filename}_power_detailed.png"
        plt.savefig(cv_plot_path)
        plt.close()
        
        print(f"Power plots saved to {plot_dir}")
        print(f"  - All channels power: {power_plot_path}")
        print(f"  - Detailed metrics: {cv_plot_path}")