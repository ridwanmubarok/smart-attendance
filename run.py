#!/usr/bin/env python
"""
Script to run the Flask application
"""
import os
import subprocess
import sys

def setup_environment():
    """Setup the environment before running the app"""
    # Create required directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/logs", exist_ok=True)
    os.makedirs("data/faces", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    os.makedirs("static/uploads", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    
    # Check if in virtual environment
    in_venv = sys.prefix != sys.base_prefix
    
    if not in_venv:
        print("WARNING: You are not running in a virtual environment!")
        response = input("Do you want to create and activate one? (y/n): ")
        
        if response.lower() == 'y':
            if not os.path.exists("venv"):
                print("Creating virtual environment...")
                subprocess.run([sys.executable, "-m", "venv", "venv"])
            
            # Provide instructions to activate venv
            if os.name == 'nt':  # Windows
                print("\nTo activate the virtual environment, run:")
                print("venv\\Scripts\\activate")
            else:  # macOS/Linux
                print("\nTo activate the virtual environment, run:")
                print("source venv/bin/activate")
            
            print("\nThen run this script again")
            return False
    
    # Check requirements
    try:
        import flask
        import torch
        import transformers
        import cv2
    except ImportError as e:
        print(f"ERROR: Missing required packages. {e}")
        print("Installing dependencies...")
        
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("\nDependencies installed. Please run the script again.")
        return False
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"GPU detected: {device_name}")
            print("CUDA is available! Model will use GPU acceleration by default.")
        else:
            print("No GPU detected. Model will run on CPU (slower).")
            print("If you have a NVIDIA GPU, make sure CUDA is properly installed.")
    except Exception as e:
        print(f"Error checking GPU: {e}")
    
    return True

def main():
    """Main function to run the app"""
    if not setup_environment():
        return
    
    print("Starting Face Attendance System with DETR...")
    print("This will download the DETR model from HuggingFace on first run (approx. 160MB)")
    
    try:
        # Run the Flask app
        from app import app
        
        # Get port from environment or use default
        port = int(os.environ.get("PORT", 5000))
        
        # Run the app
        app.run(host='0.0.0.0', port=port, debug=True)
        
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 