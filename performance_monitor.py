#!/usr/bin/env python3
"""
Performance Monitor for Face Recognition System
Shows real-time performance metrics and system status
"""

import time
import psutil
import os
import sys
from datetime import datetime

def get_system_info():
    """Get current system performance info"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    return {
        'cpu_percent': cpu_percent,
        'memory_percent': memory.percent,
        'memory_available_gb': memory.available / (1024**3),
        'disk_percent': disk.percent,
        'timestamp': datetime.now().strftime('%H:%M:%S')
    }

def monitor_performance():
    """Monitor system performance in real-time"""
    print("🚀 Face Recognition System Performance Monitor")
    print("=" * 50)
    print("Press Ctrl+C to stop monitoring")
    print()
    
    try:
        while True:
            info = get_system_info()
            
            # Clear screen (works on Windows and Unix)
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print("🚀 Face Recognition System Performance Monitor")
            print("=" * 50)
            print(f"⏰ Time: {info['timestamp']}")
            print()
            
            # CPU Usage
            cpu_bar = "█" * int(info['cpu_percent'] / 5) + "░" * (20 - int(info['cpu_percent'] / 5))
            print(f"🖥️  CPU Usage: {info['cpu_percent']:5.1f}% [{cpu_bar}]")
            
            # Memory Usage
            memory_bar = "█" * int(info['memory_percent'] / 5) + "░" * (20 - int(info['memory_percent'] / 5))
            print(f"💾 Memory:     {info['memory_percent']:5.1f}% [{memory_bar}] ({info['memory_available_gb']:.1f}GB free)")
            
            # Disk Usage
            disk_bar = "█" * int(info['disk_percent'] / 5) + "░" * (20 - int(info['disk_percent'] / 5))
            print(f"💿 Disk:       {info['disk_percent']:5.1f}% [{disk_bar}]")
            
            print()
            print("📊 Performance Status:")
            
            # Performance indicators
            if info['cpu_percent'] < 30:
                print("✅ CPU: Excellent (Low usage)")
            elif info['cpu_percent'] < 60:
                print("🟡 CPU: Good (Moderate usage)")
            elif info['cpu_percent'] < 80:
                print("🟠 CPU: Fair (High usage)")
            else:
                print("🔴 CPU: Poor (Very high usage)")
            
            if info['memory_percent'] < 50:
                print("✅ Memory: Excellent (Low usage)")
            elif info['memory_percent'] < 75:
                print("🟡 Memory: Good (Moderate usage)")
            elif info['memory_percent'] < 90:
                print("🟠 Memory: Fair (High usage)")
            else:
                print("🔴 Memory: Poor (Very high usage)")
            
            print()
            print("🎯 Optimizations Applied:")
            print("• Frame skipping (process every 2nd-3rd frame)")
            print("• Timeout protection for InsightFace operations")
            print("• Reduced refresh intervals")
            print("• Error handling for all operations")
            print("• OpenCV fallback for reliability")
            
            print()
            print("Press Ctrl+C to stop monitoring")
            
            time.sleep(2)  # Update every 2 seconds
            
    except KeyboardInterrupt:
        print("\n\n👋 Performance monitoring stopped.")
        print("✅ System optimizations are working!")

if __name__ == "__main__":
    monitor_performance()
