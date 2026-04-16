#!/usr/bin/env python3
"""
CPU Setup and Verification Script for Jetson Orin
Checks current CPU configuration and optionally sets performance mode
"""

import sys
import argparse
from cpu_freq_manager import CPUFrequencyManager

def display_cpu_status(cpu_manager):
    """Display comprehensive CPU status information"""
    print("🔍 Current CPU Configuration:")
    print("=" * 80)
    
    cpu_info = cpu_manager.get_cpu_info()
    
    for cpu_name, info in cpu_info.items():
        print(f"\n📱 {cpu_name.upper()} Status:")
        print(f"  Current Frequency: {info['current_freq']/1000:.0f} MHz" if info['current_freq'] else "  Current Frequency: Unknown")
        print(f"  Maximum Frequency: {info['max_freq']/1000:.0f} MHz" if info['max_freq'] else "  Maximum Frequency: Unknown")
        print(f"  Current Governor: {info['governor']}" if info['governor'] else "  Current Governor: Unknown")
        
        if info['available_governors']:
            print(f"  Available Governors: {', '.join(info['available_governors'])}")
        
        if info['available_frequencies']:
            freqs = info['available_frequencies']
            print(f"  Available Frequencies: {freqs[0]/1000:.0f} - {freqs[-1]/1000:.0f} MHz ({len(freqs)} steps)")
            print(f"  Max Available: {max(freqs)/1000:.0f} MHz")
    
    print("=" * 80)

def check_performance_mode(cpu_manager):
    """Check if system is already in performance mode"""
    print("\n🎯 Performance Mode Check:")
    
    cpu_info = cpu_manager.get_cpu_info()
    performance_ready = True
    
    for cpu_name, info in cpu_info.items():
        governor_ok = info['governor'] == 'performance'
        freq_ok = info['current_freq'] and info['current_freq'] >= 1900000  # Within 100MHz of max
        
        print(f"  {cpu_name}: Governor={'✅' if governor_ok else '❌'} ({info['governor']}) | "
              f"Frequency={'✅' if freq_ok else '❌'} ({info['current_freq']/1000:.0f} MHz)")
        
        if not (governor_ok and freq_ok):
            performance_ready = False
    
    if performance_ready:
        print("✅ System is already in performance mode!")
    else:
        print("⚠️ System needs performance mode configuration")
    
    return performance_ready

def set_performance_mode(cpu_manager, target_freq=1971200):
    """Set system to performance mode"""
    print(f"\n🚀 Setting Performance Mode (Target: {target_freq/1000:.0f} MHz):")
    print("-" * 60)
    
    if not cpu_manager.has_permissions:
        print("❌ Cannot set performance mode without proper permissions!")
        print("💡 Try running with sudo: sudo python cpu_setup_verify.py --set-performance")
        return False
    
    success = cpu_manager.configure_cpu_performance(target_freq)
    
    if success:
        print("\n✅ Performance mode configured successfully!")
    else:
        print("\n⚠️ Performance mode configuration completed with warnings")
    
    return success

def restore_original_settings(cpu_manager):
    """Restore original CPU settings"""
    print("\n🔄 Restoring Original Settings:")
    print("-" * 40)
    
    if not cpu_manager.has_permissions:
        print("❌ Cannot restore settings without proper permissions!")
        return False
    
    cpu_manager.restore_original_settings()
    print("✅ Original settings restored")
    return True

def main():
    parser = argparse.ArgumentParser(description='CPU Setup and Verification for Jetson Orin')
    parser.add_argument('--set-performance', action='store_true',
                       help='Set CPU to performance mode (requires sudo)')
    parser.add_argument('--restore', action='store_true',
                       help='Restore original CPU settings (requires sudo)')
    parser.add_argument('--freq', type=int, default=1971200,
                       help='Target CPU frequency in Hz (default: 1971200)')
    parser.add_argument('--check-only', action='store_true',
                       help='Only check current status, no changes')
    
    args = parser.parse_args()
    
    print("🖥️ Jetson Orin CPU Configuration Tool")
    print("=" * 50)
    
    # Initialize CPU manager
    cpu_manager = CPUFrequencyManager()
    
    # Always display current status
    display_cpu_status(cpu_manager)
    
    # Check if already in performance mode
    is_performance = check_performance_mode(cpu_manager)
    
    if args.check_only:
        print("\n📋 Check completed. Use --set-performance to configure performance mode.")
        return 0
    
    if args.restore:
        if restore_original_settings(cpu_manager):
            print("\n📊 Updated Status:")
            display_cpu_status(cpu_manager)
        return 0
    
    if args.set_performance:
        if not is_performance:
            if set_performance_mode(cpu_manager, args.freq):
                print("\n📊 Updated Status:")
                display_cpu_status(cpu_manager)
                
                print("\n🎉 Ready for CPU benchmarking!")
                print("💡 You can now run: python cpu_comprehensive_benchmark.py")
            else:
                print("\n❌ Failed to configure performance mode")
                return 1
        else:
            print("\n✅ System already in optimal performance mode!")
    else:
        if not is_performance:
            print("\n💡 To set performance mode, run:")
            print("   sudo python cpu_setup_verify.py --set-performance")
        
        print("\n💡 Available commands:")
        print("   --set-performance    Set CPU to performance mode")
        print("   --restore           Restore original settings") 
        print("   --check-only        Only check current status")
        print("   --freq FREQ         Set custom frequency (Hz)")
    
    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)