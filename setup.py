"""
Quick Setup and Validation Script
Run this to check if everything is configured correctly
"""

import os
import sys
import subprocess
import platform

def print_header(text):
    print("\n" + "="*70)
    print(text)
    print("="*70)

def check_python_version():
    """Check if Python version is compatible"""
    print("\n📦 Checking Python Version...")
    version = sys.version_info
    print(f"   Current: Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and 8 <= version.minor <= 12:
        print("   ✅ Python version is compatible")
        return True
    else:
        print(f"   ⚠️  Recommended: Python 3.8-3.12")
        print(f"   Your version may have compatibility issues")
        return False

def check_ollama_installed():
    """Check if Ollama is installed"""
    print("\n🤖 Checking Ollama Installation...")
    
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print(f"   ✅ Ollama installed: {result.stdout.strip()}")
            return True
        else:
            print("   ❌ Ollama not found")
            return False
    except FileNotFoundError:
        print("   ❌ Ollama not installed")
        print("\n   📥 Install Ollama:")
        if platform.system() == "Windows":
            print("      Download: https://ollama.ai/download/windows")
        elif platform.system() == "Darwin":
            print("      Download: https://ollama.ai/download/mac")
            print("      Or: brew install ollama")
        else:
            print("      Run: curl -fsSL https://ollama.ai/install.sh | sh")
        return False
    except Exception as e:
        print(f"   ⚠️  Error checking Ollama: {e}")
        return False

def check_ollama_model(model_name="llama3.1:8b"):
    """Check if required model is downloaded"""
    print(f"\n📚 Checking Model: {model_name}...")
    
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if model_name in result.stdout:
            print(f"   ✅ Model '{model_name}' is downloaded")
            return True
        else:
            print(f"   ❌ Model '{model_name}' not found")
            print(f"\n   📥 Download the model:")
            print(f"      ollama pull {model_name}")
            return False
            
    except Exception as e:
        print(f"   ⚠️  Error checking models: {e}")
        return False

def check_required_files():
    """Check if required data files exist"""
    print("\n📄 Checking Required Files...")
    
    required_files = {
        'train.csv': 'Training data',
        'test.csv': 'Test data',
        'The_Count_of_Monte_Cristo.txt': 'Novel 1',
        'In_search_of_the_castaways.txt': 'Novel 2'
    }
    
    all_present = True
    for filename, description in required_files.items():
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"   ✅ {description}: {filename} ({size:,} bytes)")
        else:
            print(f"   ❌ {description}: {filename} (NOT FOUND)")
            all_present = False
    
    return all_present

def check_dependencies():
    """Check if Python packages are installed"""
    print("\n📦 Checking Python Dependencies...")
    
    required = {
        'ollama': 'Ollama Python client',
        'pandas': 'Data processing',
        'numpy': 'Numerical operations',
        'sentence_transformers': 'Embeddings',
        'sklearn': 'Machine learning utilities',
        'dotenv': 'Environment variables'
    }
    
    missing = []
    
    for package, description in required.items():
        try:
            __import__(package)
            print(f"   ✅ {description}: {package}")
        except ImportError:
            print(f"   ❌ {description}: {package} (NOT INSTALLED)")
            missing.append(package)
    
    if missing:
        print("\n   📥 Install missing packages:")
        print("      pip install -r requirements.txt")
        print("\n   Or install individually:")
        for pkg in missing:
            print(f"      pip install {pkg}")
        return False
    
    return True

def test_ollama_connection(model_name="llama3.1:8b"):
    """Test if Ollama is actually working"""
    print(f"\n🔌 Testing Ollama Connection...")
    
    try:
        import ollama
        
        response = ollama.generate(
            model=model_name,
            prompt="Say 'OK' if you're working.",
            options={'num_predict': 10}
        )
        
        print(f"   ✅ Ollama is working!")
        print(f"   Response: {response['response'][:50]}...")
        return True
        
    except Exception as e:
        print(f"   ❌ Ollama connection failed: {e}")
        print("\n   💡 Troubleshooting:")
        print("      1. Make sure Ollama is running")
        print("      2. Try: ollama serve")
        print(f"      3. Verify model: ollama list")
        return False

def create_env_file():
    """Create .env template if it doesn't exist"""
    if not os.path.exists('.env'):
        print("\n📝 Creating .env file...")
        with open('.env', 'w') as f:
            f.write("# Ollama Configuration\n")
            f.write("OLLAMA_MODEL=llama3.1:8b\n")
            f.write("SKIP_TRAINING_EVAL=false\n")
            f.write("\n# Set to 'true' to skip training evaluation (faster)\n")
        print("   ✅ Created .env file")
    else:
        print("\n📝 .env file already exists")

def main():
    """Run all checks"""
    
    print_header("🚀 Narrative Consistency Checker - Setup Validation")
    
    print("\nOS:", platform.system(), platform.release())
    
    checks = {
        'Python Version': check_python_version(),
        'Ollama Installed': check_ollama_installed(),
    }
    
    # Only check model if Ollama is installed
    if checks['Ollama Installed']:
        model_name = os.getenv('OLLAMA_MODEL', 'llama3.1:8b')
        checks['Model Downloaded'] = check_ollama_model(model_name)
    else:
        checks['Model Downloaded'] = False
    
    checks['Required Files'] = check_required_files()
    checks['Dependencies'] = check_dependencies()
    
    # Test connection if everything else passes
    if all([checks['Ollama Installed'], checks['Model Downloaded'], checks['Dependencies']]):
        model_name = os.getenv('OLLAMA_MODEL', 'llama3.1:8b')
        checks['Ollama Connection'] = test_ollama_connection(model_name)
    else:
        checks['Ollama Connection'] = False
    
    # Create .env file
    create_env_file()
    
    # Summary
    print_header("📊 SETUP SUMMARY")
    
    all_good = True
    for check_name, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check_name}")
        if not passed:
            all_good = False
    
    print("\n" + "="*70)
    
    if all_good:
        print("✅ ALL CHECKS PASSED!")
        print("\n🚀 You're ready to run:")
        print("   python main.py")
    else:
        print("❌ SOME CHECKS FAILED")
        print("\n🔧 Please fix the issues above before running main.py")
        print("\n📖 See README.md for detailed setup instructions:")
        print("   - Ollama installation: README.md#installation")
        print("   - Model download: README.md#step-2-download-llm-model")
        print("   - Data preparation: README.md#step-4-prepare-data-files")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    main()