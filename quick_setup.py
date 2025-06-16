#!/usr/bin/env python3
"""
Quick Setup Script for Knowledge Base MCP Server
Creates necessary directories and configuration files
"""

import os
from pathlib import Path

def safe_print(text):
    """Print with fallback for systems that can't display Unicode"""
    try:
        print(text)
    except UnicodeEncodeError:
        # Fallback: remove emojis and use ASCII
        fallback_text = text.encode('ascii', 'ignore').decode('ascii')
        print(fallback_text)

def create_directories():
    """Create necessary directories with proper path joining."""
    safe_print("Creating Knowledge Base directories...")
    
    directories = [
        os.path.join(".", "knowledge_base"),
        os.path.join(".", "knowledge_base", "pdfs"),
        os.path.join(".", "knowledge_base", "research_papers"),
        os.path.join(".", "knowledge_base", "textbooks"),
        os.path.join(".", "knowledge_base", "vector_db")
    ]
    
    created = []
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            created.append(directory)
    
    if created:
        safe_print(f"Created directories: {', '.join(created)}")
    else:
        safe_print("All directories already exist")

def create_sample_env():
    """Create a sample .env file if one doesn't exist."""
    env_file = Path(".env")
    
    if env_file.exists():
        safe_print(".env file already exists")
        return
    
    # Get current working directory for sample paths
    current_dir = Path.cwd()
    
    sample_env_content = f"""# Knowledge Base MCP Server Configuration
# Copy and modify these settings for your environment

# Logging
LOG_LEVEL=INFO
TIMEZONE=UTC

# Obsidian Configuration - UPDATE THESE PATHS!
OBSIDIAN_VAULT_PATH={os.path.join(str(current_dir), "sample_vault")}
OBSIDIAN_NOTES_FOLDER=Knowledge_Base_Notes
OBSIDIAN_TEMPLATES_FOLDER=Templates
OBSIDIAN_ATTACHMENTS_FOLDER=Attachments
OBSIDIAN_DEFAULT_TEMPLATE=
OBSIDIAN_USE_DAILY_NOTES=false

# PDF Knowledge Base Configuration
PDF_DIRECTORY_1={os.path.join(str(current_dir), "knowledge_base", "pdfs")}
PDF_DIRECTORY_2={os.path.join(str(current_dir), "knowledge_base", "research_papers")}
PDF_DIRECTORY_3={os.path.join(str(current_dir), "knowledge_base", "textbooks")}

# Vector Database Configuration
VECTOR_DB_PATH={os.path.join(str(current_dir), "knowledge_base", "vector_db")}
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Advanced LangChain Chunking Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# LangChain Chunking Strategy Options:
# - auto: Automatically select best strategy based on content
# - recursive: RecursiveCharacterTextSplitter (best for general text)
# - token: TokenTextSplitter (precise token control)
# - character: CharacterTextSplitter (simple paragraph splitting)
# - markdown: MarkdownTextSplitter (for structured content)
CHUNKING_STRATEGY=auto
USE_SEMANTIC_CHUNKING=true
PRESERVE_CHUNK_BOUNDARIES=true

# Token-based chunking (when using token strategy)
MAX_TOKENS_PER_CHUNK=250
TOKEN_OVERLAP=50

# Content-aware chunking
DETECT_CONTENT_TYPE=true
SPLIT_ON_SENTENCES=true
SPLIT_ON_PARAGRAPHS=true

# Search Settings
MAX_SEARCH_RESULTS=10
SIMILARITY_THRESHOLD=0.7

# Content Processing Settings
EXTRACT_IMAGES=false
EXTRACT_TABLES=true
PRESERVE_FORMATTING=true
MAX_NOTE_LENGTH=5000
INCLUDE_CITATIONS=true
AUTO_GENERATE_TAGS=true

# Learning Features
GENERATE_FLASHCARDS=true
CREATE_MIND_MAPS=false
TRACK_LEARNING_PROGRESS=true

# Performance Settings
MAX_CONCURRENT_PROCESSES=3
"""
    
    with open(env_file, 'w', encoding='utf-8') as f:
        f.write(sample_env_content)
    
    safe_print(f"Created sample .env file: {env_file}")
    safe_print("Please edit the .env file with your actual paths before running the server")

def main():
    """Run the quick setup."""
    safe_print("Knowledge Base MCP Server - Quick Setup")
    safe_print("=" * 50)
    
    try:
        create_directories()
        create_sample_env()
        
        safe_print("\nSetup complete!")
        safe_print("\nNext steps:")
        safe_print("1. Edit the .env file with your actual Obsidian vault path")
        safe_print("2. Add some PDF files to the knowledge_base/pdfs directory")
        safe_print("3. Run: python run_mcp_server.py")
        
    except Exception as e:
        safe_print(f"Error during setup: {e}")

if __name__ == "__main__":
    main() 