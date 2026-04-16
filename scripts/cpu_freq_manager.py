#!/usr/bin/env python3
"""
CPU Frequency Manager for Jetson Orin platforms
Manages CPU frequency scaling for performance benchmarking
"""

import os
import time
from pathlib import Path

class CPUFrequencyManager:
    def __init__(self):
        # CPU paths for Jetson Orin (typically cpu0 and cpu4 are the main cores)
        self.cpu_paths = [
            "/sys/devices/system/cpu/cpu0/cpufreq",
            "/sys/devices/system/cpu/cpu4/cpufreq"
        ]
        self.original_frequencies = {}
        self.original_governors = {}
        self.has_permissions = self._check_permissions()
        
    def _check_permissions(self):
        """Check if we have permissions to modify CPU frequency"""
        try:
            for cpu_path in self.cpu_paths:
                with open(f"{cpu_path}/scaling_max_freq", 'r') as f:
                    _ = f.read()
            return True
        except Exception as e:
            print(f"Note: Running without CPU frequency management (requires sudo for writes): {e}")
            return False
            
    def read_current_freq(self, cpu_path):
        """Read current CPU frequency for a specific CPU"""
        try:
            with open(f"{cpu_path}/scaling_cur_freq", 'r') as f:
                return int(f.read().strip())
        except Exception as e:
            print(f"Note: Could not read CPU frequency from {cpu_path}: {e}")
            return None
            
    def read_max_freq(self, cpu_path):
        """Read maximum CPU frequency for a specific CPU"""
        try:
            with open(f"{cpu_path}/scaling_max_freq", 'r') as f:
                return int(f.read().strip())
        except Exception as e:
            print(f"Note: Could not read max CPU frequency from {cpu_path}: {e}")
            return None
            
    def read_governor(self, cpu_path):
        """Read current CPU governor for a specific CPU"""
        try:
            with open(f"{cpu_path}/scaling_governor", 'r') as f:
                return f.read().strip()
        except Exception as e:
            print(f"Note: Could not read CPU governor from {cpu_path}: {e}")
            return None
            
    def set_governor(self, cpu_path, governor):
        """Set CPU governor for a specific CPU"""
        if not self.has_permissions:
            return False
            
        try:
            with open(f"{cpu_path}/scaling_governor", 'w') as f:
                f.write(governor)
            return True
        except Exception as e:
            print(f"Note: Could not set CPU governor for {cpu_path}: {e}")
            return False
            
    def set_max_freq(self, cpu_path, freq):
        """Set maximum CPU frequency for a specific CPU"""
        if not self.has_permissions:
            return False
            
        try:
            with open(f"{cpu_path}/scaling_max_freq", 'w') as f:
                f.write(str(freq))
            return True
        except Exception as e:
            print(f"Note: Could not set max CPU frequency for {cpu_path}: {e}")
            return False
            
    def set_min_freq(self, cpu_path, freq):
        """Set minimum CPU frequency for a specific CPU"""
        if not self.has_permissions:
            return False
            
        try:
            with open(f"{cpu_path}/scaling_min_freq", 'w') as f:
                f.write(str(freq))
            return True
        except Exception as e:
            print(f"Note: Could not set min CPU frequency for {cpu_path}: {e}")
            return False
            
    def configure_cpu_performance(self, target_freq=1971200):
        """Configure CPU for maximum performance at specified frequency"""
        if not self.has_permissions:
            print("Note: Cannot configure CPU frequency without proper permissions")
            return False
            
        print(f"Configuring CPU performance mode at {target_freq} Hz...")
        
        success = True
        for i, cpu_path in enumerate(self.cpu_paths):
            cpu_name = f"cpu{0 if i == 0 else 4}"
            
            # Store original settings
            self.original_frequencies[cpu_path] = self.read_max_freq(cpu_path)
            self.original_governors[cpu_path] = self.read_governor(cpu_path)
            
            print(f"  {cpu_name}: Original governor: {self.original_governors[cpu_path]}, "
                  f"max freq: {self.original_frequencies[cpu_path]} Hz")
            
            # STEP 1: Set performance governor for consistent frequency
            print(f"    Setting governor to 'performance' for {cpu_name}...")
            if not self.set_governor(cpu_path, "performance"):
                print(f"    ❌ Could not set performance governor for {cpu_name}")
                success = False
                continue
            
            # Verify governor was set
            time.sleep(0.1)
            current_governor = self.read_governor(cpu_path)
            if current_governor == "performance":
                print(f"    ✅ Governor set to 'performance' for {cpu_name}")
            else:
                print(f"    ⚠️ Governor verification failed for {cpu_name}: expected 'performance', got '{current_governor}'")
                success = False
                
            # STEP 2: Set both min and max frequency to target frequency for fixed frequency
            print(f"    Setting frequency to {target_freq} Hz for {cpu_name}...")
            if not self.set_min_freq(cpu_path, target_freq):
                print(f"    ⚠️ Could not set min frequency for {cpu_name}")
                success = False
                
            if not self.set_max_freq(cpu_path, target_freq):
                print(f"    ⚠️ Could not set max frequency for {cpu_name}")
                success = False
                
            # STEP 3: Verify the frequency was set
            time.sleep(0.2)  # Allow time for frequency change
            current_freq = self.read_current_freq(cpu_path)
            max_freq = self.read_max_freq(cpu_path)
            
            print(f"    {cpu_name}: Target: {target_freq} Hz, Current: {current_freq} Hz, Max: {max_freq} Hz")
            
            # Check if frequency is close to target (within 5%)
            if current_freq and abs(current_freq - target_freq) / target_freq < 0.05:
                print(f"    ✅ Frequency successfully set for {cpu_name}")
            else:
                print(f"    ⚠️ Frequency may not be at target for {cpu_name}")
            
        if success:
            print("✅ CPU performance configuration completed successfully")
        else:
            print("⚠️ CPU performance configuration completed with warnings")
            
        # Final verification - show current status
        print("\n📊 Current CPU Status:")
        cpu_info = self.get_cpu_info()
        for cpu_name, info in cpu_info.items():
            print(f"  {cpu_name}: {info['current_freq']/1000:.0f} MHz ({info['governor']})")
            
        return success
        
    def restore_original_settings(self):
        """Restore original CPU frequency and governor settings"""
        if not self.has_permissions:
            return
            
        print("Restoring original CPU frequency settings...")
        
        for i, cpu_path in enumerate(self.cpu_paths):
            cpu_name = f"cpu{0 if i == 0 else 4}"
            
            if cpu_path in self.original_governors:
                # First restore governor
                original_gov = self.original_governors[cpu_path]
                if self.set_governor(cpu_path, original_gov):
                    print(f"  {cpu_name}: Restored governor to {original_gov}")
                else:
                    print(f"  {cpu_name}: Warning - Could not restore governor")
                    
            if cpu_path in self.original_frequencies:
                # Then restore max frequency
                original_freq = self.original_frequencies[cpu_path]
                if self.set_max_freq(cpu_path, original_freq):
                    print(f"  {cpu_name}: Restored max frequency to {original_freq} Hz")
                else:
                    print(f"  {cpu_name}: Warning - Could not restore max frequency")
                    
                # Reset min frequency to a reasonable default
                self.set_min_freq(cpu_path, 115200)  # Minimum available frequency
                
        # Clear stored settings
        self.original_frequencies.clear()
        self.original_governors.clear()
        print("🔄 CPU frequency settings restored")
        
    def read_available_governors(self, cpu_path):
        """Read available CPU governors for a specific CPU"""
        try:
            with open(f"{cpu_path}/scaling_available_governors", 'r') as f:
                return f.read().strip().split()
        except Exception as e:
            print(f"Note: Could not read available governors from {cpu_path}: {e}")
            return []
            
    def read_available_frequencies(self, cpu_path):
        """Read available CPU frequencies for a specific CPU"""
        try:
            with open(f"{cpu_path}/scaling_available_frequencies", 'r') as f:
                return [int(freq) for freq in f.read().strip().split()]
        except Exception as e:
            print(f"Note: Could not read available frequencies from {cpu_path}: {e}")
            return []
            
    def get_cpu_info(self):
        """Get current CPU frequency information"""
        info = {}
        for i, cpu_path in enumerate(self.cpu_paths):
            cpu_name = f"cpu{0 if i == 0 else 4}"
            info[cpu_name] = {
                'current_freq': self.read_current_freq(cpu_path),
                'max_freq': self.read_max_freq(cpu_path),
                'governor': self.read_governor(cpu_path),
                'available_governors': self.read_available_governors(cpu_path),
                'available_frequencies': self.read_available_frequencies(cpu_path)
            }
        return info