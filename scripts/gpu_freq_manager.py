class GPUFrequencyManager:
    def __init__(self):
        self.gpu_path = "/sys/devices/gpu.0/devfreq/17000000.ga10b"
        self.original_max_freq = None
        self.has_permissions = self._check_permissions()
        
    def _check_permissions(self):
        """Check if we have permissions to modify GPU frequency"""
        try:
            with open(f"{self.gpu_path}/max_freq", 'r') as f:
                _ = f.read()
            return True
        except Exception:
            print("Note: Running without GPU frequency management (requires sudo for writes)")
            return False
            
    def read_current_max_freq(self):
        """Read current maximum GPU frequency"""
        try:
            with open(f"{self.gpu_path}/max_freq", 'r') as f:
                return int(f.read().strip())
        except Exception as e:
            print(f"Note: Could not read GPU frequency: {e}")
            return None
            
    def set_max_freq(self, freq):
        """Set maximum GPU frequency if we have permissions"""
        if not self.has_permissions:
            return False
            
        try:
            if self.original_max_freq is None:
                self.original_max_freq = self.read_current_max_freq()
                print(f"Original GPU max frequency: {self.original_max_freq} Hz")
            
            # Try to set frequency directly (will work if we have permissions)
            with open(f"{self.gpu_path}/max_freq", 'w') as f:
                f.write(str(freq))
            
            new_freq = self.read_current_max_freq()
            if new_freq != freq:
                print(f"Note: Could not set GPU frequency. Current: {new_freq} Hz, Requested: {freq} Hz")
            else:
                print(f"Successfully set GPU max frequency to: {new_freq} Hz")
            
            return True
        except Exception as e:
            print(f"Note: Could not set GPU frequency: {e}")
            return False
            
    def restore_original_freq(self):
        """Restore original GPU frequency if we have permissions"""
        if self.has_permissions and self.original_max_freq is not None:
            success = self.set_max_freq(self.original_max_freq)
            if success:
                print(f"Restored GPU max frequency to: {self.original_max_freq} Hz")
            self.original_max_freq = None