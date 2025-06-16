#!/usr/bin/env python3
"""
Run Knowledge Base MCP Server for Cursor Integration
This script starts the MCP server that Cursor can connect to
"""

import asyncio
import sys
import logging
import os
from pathlib import Path

# Fix Unicode encoding issues on Windows
if sys.platform == "win32":
    import codecs
    # Set console encoding to UTF-8
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    # Set environment variable for UTF-8
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from server import main

def safe_print(text):
    """Print with fallback for systems that can't display Unicode"""
    try:
        print(text)
    except UnicodeEncodeError:
        # Fallback: remove emojis and use ASCII
        fallback_text = text.encode('ascii', 'ignore').decode('ascii')
        print(fallback_text)

if __name__ == "__main__":
    # Only print startup messages if explicitly enabled for debugging
    # MCP servers should not print to stdout as it interferes with JSON-RPC
    debug_mode = os.getenv("KB_DEBUG", "false").lower() == "true"
    
    if debug_mode:
        safe_print("🚀 Starting Knowledge Base MCP Server for Cursor")
        safe_print("=" * 60)
        safe_print("📡 Server Status: Starting...")
        safe_print("🔗 Protocol: Model Context Protocol (MCP)")
        safe_print("🎯 Client: Cursor IDE")
        safe_print("📚 Knowledge Base: PDF Vector Database")
        safe_print("📝 Output: Obsidian Notes")
        safe_print("=" * 60)
    
    try:
        # Run the MCP server
        asyncio.run(main())
    except KeyboardInterrupt:
        if debug_mode:
            safe_print("\n🛑 Server stopped by user")
        sys.exit(0)
    except Exception as e:
        if debug_mode:
            safe_print(f"\n❌ Server error: {e}")
        sys.exit(1) 