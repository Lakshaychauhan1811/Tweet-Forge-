#!/usr/bin/env python3
"""
TweetForge Pro - Setup Script
This script helps you set up the environment and run the application
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def print_banner():
    print("=" * 60)
    print("🚀 TweetForge - AI-Powered Tweet Generator")
    print("=" * 60)
    print()

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def create_env_file():
    """Create .env file from template if it doesn't exist"""
    env_file = Path(".env")
    env_example = Path("env_example.txt")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    if env_example.exists():
        shutil.copy(env_example, env_file)
        print("✅ Created .env file from template")
        print("⚠️  Please edit .env file and add your API keys")
        return True
    else:
        print("❌ env_example.txt not found")
        return False

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_api_keys():
    """Check if API keys are configured"""
    from dotenv import load_dotenv 
    load_dotenv()
    
    groq_key = os.getenv('GROQ_API_KEY')
    serper_key = os.getenv('SERPER_API_KEY')
    
    print("\n🔑 API Key Status:")
    if groq_key and groq_key != 'your_groq_api_key_here':
        print("✅ GROQ API key configured")
    else:
        print("⚠️  GROQ API key not configured (required for AI features)")
        print("   Get your key from: https://console.groq.com/")
    
    if serper_key and serper_key != 'your_serper_api_key_here':
        print("✅ Serper API key configured")
    else:
        print("⚠️  Serper API key not configured (optional for internet research)")
        print("   Get your key from: https://serper.dev/")
    
    return groq_key and groq_key != 'your_groq_api_key_here'

def run_server():
    """Start the server"""
    print("\n🚀 Starting TweetForge server...")
    print("📱 Open your browser and go to: http://127.0.0.1:8000")
    print("🔄 Press Ctrl+C to stop the server")
    print()
    
    try:
        subprocess.run([sys.executable, "start.py"])
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

def main():
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create .env file
    if not create_env_file():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("⚠️  Continuing without installing dependencies...")
    
    # Check API keys
    has_groq_key = check_api_keys()
    
    if not has_groq_key:
        print("\n⚠️  Warning: GROQ API key not configured")
        print("   The app will work with basic templates instead of AI-generated content")
        print("   To enable AI features, edit .env file and add your GROQ API key")
    
    print("\n" + "=" * 60)
    print("🎉 Setup complete! Ready to start TweetForge")
    print("=" * 60)
    
    # Ask if user wants to start the server
    response = input("\n🚀 Start the server now? (y/n): ").lower().strip()
    if response in ['y', 'yes', '']:
        run_server()
    else:
        print("\n📝 To start the server later, run:")
        print("   python start.py")
        print("   or")
        print("   python setup.py")

if __name__ == "__main__":
    main()
