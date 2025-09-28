#!/usr/bin/env python3
"""
Simple launcher for TweetForge
Just run: python run.py
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("🚀 Starting TweetForge...")
    
    # Check if we're in the right directory
    if not Path("start.py").exists():
        print("❌ start.py not found! Make sure you're in the project directory.")
        sys.exit(1)
    
    # Run start.py
    try:
        subprocess.run([sys.executable, "start.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 TweetForge stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
