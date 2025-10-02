#!/usr/bin/env python3


import os
import sys
import subprocess
from pathlib import Path

def run_with_venv():
    """Run the script using a local virtual environment's Python.
    Prefer `venv/` but fall back to `env/` if present.
    """
    project_dir = Path(__file__).parent
    candidates = [
        project_dir / "venv" / "Scripts" / "python.exe",
        project_dir / "env" / "Scripts" / "python.exe",
    ]
    venv_python = next((p for p in candidates if p.exists()), None)

    if venv_python is None:
        print("❌ Virtual environment not found!")
        print("   Create one and install deps:")
        print("   python -m venv venv && .\\venv\\Scripts\\python.exe -m pip install -r requirements.txt")
        return False

    # Run this same script with the virtual environment Python
    try:
        subprocess.run([str(venv_python), __file__] + sys.argv[1:], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running with virtual environment: {e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
        return True

# Check if we're already running inside the project's venv
exe_path = Path(sys.executable)
if ("venv" in exe_path.as_posix()) or ("env" in exe_path.as_posix() and "Scripts" in exe_path.as_posix()):
    # We're in the virtual environment, proceed normally
    try:
        import uvicorn
        from dotenv import load_dotenv
    except ImportError as e:
        print(f"❌ Missing required packages: {e}")
        print("   Please run: .\\venv\\Scripts\\python.exe -m pip install -r requirements.txt")
        sys.exit(1)
else:
    # We're not in the virtual environment, redirect to venv
    if not run_with_venv():
        sys.exit(1)
    sys.exit(0)

def check_environment():
    """Check if required environment variables are set"""
    load_dotenv()
    
    print("🔍 Checking environment setup...")
    
    # Check if .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found!")
        print("   Please copy env_example.txt to .env and configure your API keys")
        print("   Or run: python setup.py")
        return False
    
    # Check for GROQ API key
    groq_key = os.getenv('GROQ_API_KEY')
    if not groq_key or groq_key == 'your_groq_api_key_here':
        print("⚠️  Warning: GROQ_API_KEY not configured in .env file")
        print("   The app will work with basic templates instead of AI-generated content")
        print("   Get your key from: https://console.groq.com/")
    else:
        print("✅ GROQ API key configured")
    
    # Check for Serper API key
    serper_key = os.getenv('SERPER_API_KEY')
    if not serper_key or serper_key == 'your_serper_api_key_here':
        print("⚠️  Warning: SERPER_API_KEY not configured in .env file")
        print("   Internet research will be limited")
        print("   Get your key from: https://serper.dev/")
    else:
        print("✅ Serper API key configured")
    
    print()
    return True

def main():
    """Main startup function"""
    print("🚀 Starting TweetForge - AI-Powered Tweet Generator")
    print("=" * 60)
    
    # Check environment
    if not check_environment():
        print("❌ Environment check failed. Please fix the issues above.")
        sys.exit(1)
    
    # Get configuration
    host = os.getenv('HOST', '127.0.0.1')
    port = int(os.getenv('PORT', 8000))
    
    print(f"🌐 Server will start at: http://{host}:{port}")
    print("📱 Open your browser and navigate to the URL above")
    print("🔄 Press Ctrl+C to stop the server")
    print("📚 API documentation available at: http://{host}:{port}/api/docs")
    print()
    
    try:
        # Start the server
        uvicorn.run(
            "app:app",
            host=host,
            port=port,
            reload=True,  # Enable auto-reload for development
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        print("   Make sure the port is not already in use")
        sys.exit(1)

if __name__ == "__main__":
    main()
