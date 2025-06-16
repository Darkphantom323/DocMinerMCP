#!/usr/bin/env python3
"""
Dependency Installation Script for Knowledge Base MCP Server
Handles common installation issues and ensures proper package versions
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def install_pytorch_first():
    """Install PyTorch first to avoid conflicts."""
    print("\n🚀 Installing PyTorch (required for sentence-transformers)...")
    
    commands = [
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
        "pip install torch torchvision torchaudio"  # Fallback
    ]
    
    for cmd in commands:
        if run_command(cmd, "Installing PyTorch"):
            return True
    
    print("⚠️ PyTorch installation failed, continuing anyway...")
    return False

def install_dependencies():
    """Install all dependencies with proper order."""
    print("📦 Installing Knowledge Base MCP Server Dependencies")
    print("=" * 60)
    
    # Upgrade pip first
    run_command("python -m pip install --upgrade pip", "Upgrading pip")
    
    # Install PyTorch first (often fixes dependency issues)
    install_pytorch_first()
    
    # Install core dependencies individually to handle conflicts
    core_deps = [
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
    ]
    
    print("\n📋 Installing core dependencies...")
    for dep in core_deps:
        run_command(f"pip install {dep}", f"Installing {dep}")
    
    # Install PDF processing libraries
    pdf_deps = [
        "PyPDF2>=3.0.0",
        "pdfplumber>=0.9.0",
        "PyMuPDF>=1.23.0",
    ]
    
    print("\n📄 Installing PDF processing libraries...")
    for dep in pdf_deps:
        run_command(f"pip install {dep}", f"Installing {dep}")
    
    # Install text processing and embeddings
    ml_deps = [
        "sentence-transformers>=2.2.0",
        "transformers>=4.21.0",
        "chromadb>=0.4.22",
    ]
    
    print("\n🤖 Installing ML and text processing libraries...")
    for dep in ml_deps:
        run_command(f"pip install {dep}", f"Installing {dep}")
    
    # Install file processing
    file_deps = [
        "python-frontmatter>=1.0.0",
        "markdown>=3.5.0",
        "watchdog>=3.0.0",
    ]
    
    print("\n📁 Installing file processing libraries...")
    for dep in file_deps:
        run_command(f"pip install {dep}", f"Installing {dep}")
    
    # Install MCP
    print("\n🔌 Installing MCP...")
    run_command("pip install mcp", "Installing MCP")
    
    # Finally, install remaining dependencies
    remaining_deps = [
        "requests>=2.31.0",
        "python-dateutil>=2.8.0",
        "regex>=2023.0.0",
        "tiktoken>=0.5.0",
    ]
    
    print("\n🔧 Installing remaining dependencies...")
    for dep in remaining_deps:
        run_command(f"pip install {dep}", f"Installing {dep}")

def test_imports():
    """Test that all required modules can be imported."""
    print("\n🧪 Testing imports...")
    
    test_modules = [
        ("mcp", "MCP framework"),
        ("sentence_transformers", "Sentence Transformers"),
        ("chromadb", "ChromaDB"),
        ("PyPDF2", "PyPDF2"),
        ("pdfplumber", "PDF Plumber"),
        ("fitz", "PyMuPDF (fitz)"),
        ("frontmatter", "Python Frontmatter"),
        ("pydantic", "Pydantic"),
    ]
    
    success_count = 0
    for module, description in test_modules:
        try:
            __import__(module)
            print(f"✅ {description}")
            success_count += 1
        except ImportError as e:
            print(f"❌ {description}: {e}")
    
    print(f"\n📊 Import test results: {success_count}/{len(test_modules)} modules imported successfully")
    return success_count == len(test_modules)

def fix_common_issues():
    """Fix common installation issues."""
    print("\n🔨 Fixing common issues...")
    
    # Clear pip cache
    run_command("pip cache purge", "Clearing pip cache")
    
    # Update setuptools and wheel
    run_command("pip install --upgrade setuptools wheel", "Updating setuptools and wheel")
    
    # Install specific compatible versions for problematic packages
    fixes = [
        "pip install 'chromadb>=0.4.22,<0.5.0'",
        "pip install 'sentence-transformers>=2.2.0,<3.0.0'",
        "pip install 'transformers>=4.21.0,<5.0.0'",
    ]
    
    for fix in fixes:
        run_command(fix, f"Applying fix: {fix}")

def main():
    """Main installation function."""
    print("🧠 Knowledge Base MCP Server - Dependency Installer")
    print("=" * 60)
    print("This script will install all required dependencies for the Knowledge Base MCP Server.")
    print("It handles common installation issues and ensures compatibility.")
    
    # Check Python version
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ is required. Current version: {version.major}.{version.minor}.{version.micro}")
        sys.exit(1)
    
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    
    # Install dependencies
    install_dependencies()
    
    # Test imports
    if not test_imports():
        print("\n⚠️ Some imports failed. Attempting to fix common issues...")
        fix_common_issues()
        
        # Test again
        print("\n🔄 Re-testing imports after fixes...")
        if test_imports():
            print("\n🎉 All dependencies installed successfully!")
        else:
            print("\n⚠️ Some imports still failing. You may need to:")
            print("   1. Restart your terminal/IDE")
            print("   2. Check for conflicting Python environments")
            print("   3. Try installing in a virtual environment")
    else:
        print("\n🎉 All dependencies installed successfully!")
    
    print("\n📚 Next steps:")
    print("   1. Run: python setup.py")
    print("   2. Edit your .env file with correct paths")
    print("   3. Run: python test_setup.py")
    print("   4. Start using: python server.py")

if __name__ == "__main__":
    main() 