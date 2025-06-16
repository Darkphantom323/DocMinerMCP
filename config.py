"""
Configuration settings for Knowledge Base MCP Server
"""

import os
from typing import Dict, Any, List
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

class ObsidianConfig(BaseModel):
    """Obsidian vault configuration settings."""
    
    vault_path: str = os.getenv("OBSIDIAN_VAULT_PATH", "")
    notes_folder: str = os.getenv("OBSIDIAN_NOTES_FOLDER", "")  # Subfolder for generated notes
    templates_folder: str = os.getenv("OBSIDIAN_TEMPLATES_FOLDER", "Templates")
    attachments_folder: str = os.getenv("OBSIDIAN_ATTACHMENTS_FOLDER", "Attachments")
    
    # Note creation settings
    default_template: str = os.getenv("OBSIDIAN_DEFAULT_TEMPLATE", "")
    use_daily_notes: bool = os.getenv("OBSIDIAN_USE_DAILY_NOTES", "false").lower() == "true"
    auto_link_creation: bool = True
    tag_prefix: str = "kb/"  # Prefix for knowledge base tags

class PDFConfig(BaseModel):
    """PDF knowledge base configuration settings."""
    
    # Use proper path joining for cross-platform compatibility
    pdf_directories: List[str] = Field(default_factory=lambda: [
        os.getenv("PDF_DIRECTORY_1", os.path.join(".", "knowledge_base", "pdfs")),
        os.getenv("PDF_DIRECTORY_2", os.path.join(".", "knowledge_base", "research_papers")),
        os.getenv("PDF_DIRECTORY_3", os.path.join(".", "knowledge_base", "textbooks"))
    ])
    
    # Remove empty directories
    def __init__(self, **data):
        super().__init__(**data)
        self.pdf_directories = [d for d in self.pdf_directories if d]
    
    # Vector database settings - use proper path joining
    vector_db_path: str = os.getenv("VECTOR_DB_PATH", os.path.join(".", "knowledge_base", "vector_db"))
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # Advanced chunking settings with LangChain
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # LangChain specific chunking options
    chunking_strategy: str = os.getenv("CHUNKING_STRATEGY", "auto")  # auto, recursive, token, character, markdown
    use_semantic_chunking: bool = os.getenv("USE_SEMANTIC_CHUNKING", "true").lower() == "true"
    preserve_chunk_boundaries: bool = os.getenv("PRESERVE_CHUNK_BOUNDARIES", "true").lower() == "true"
    
    # Token-based chunking (for precise token control)
    max_tokens_per_chunk: int = int(os.getenv("MAX_TOKENS_PER_CHUNK", "250"))  # ~1000 chars / 4
    token_overlap: int = int(os.getenv("TOKEN_OVERLAP", "50"))  # ~200 chars / 4
    
    # Content-aware chunking
    detect_content_type: bool = os.getenv("DETECT_CONTENT_TYPE", "true").lower() == "true"
    split_on_sentences: bool = os.getenv("SPLIT_ON_SENTENCES", "true").lower() == "true"
    split_on_paragraphs: bool = os.getenv("SPLIT_ON_PARAGRAPHS", "true").lower() == "true"
    
    # Search settings - Enhanced for better results
    max_search_results: int = int(os.getenv("MAX_SEARCH_RESULTS", "15"))  # More results for multi-strategy search
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.1"))  # Lower threshold for more results

class KnowledgeBaseConfig(BaseModel):
    """Knowledge base processing configuration."""
    
    # Content extraction settings
    extract_images: bool = os.getenv("EXTRACT_IMAGES", "false").lower() == "true"
    extract_tables: bool = os.getenv("EXTRACT_TABLES", "true").lower() == "true"
    preserve_formatting: bool = os.getenv("PRESERVE_FORMATTING", "true").lower() == "true"
    
    # Note generation settings
    max_note_length: int = int(os.getenv("MAX_NOTE_LENGTH", "5000"))
    include_citations: bool = os.getenv("INCLUDE_CITATIONS", "true").lower() == "true"
    auto_generate_tags: bool = os.getenv("AUTO_GENERATE_TAGS", "true").lower() == "true"
    
    # Learning features
    generate_flashcards: bool = os.getenv("GENERATE_FLASHCARDS", "true").lower() == "true"
    create_mind_maps: bool = os.getenv("CREATE_MIND_MAPS", "false").lower() == "true"
    track_learning_progress: bool = os.getenv("TRACK_LEARNING_PROGRESS", "true").lower() == "true"

class Config(BaseModel):
    """Main configuration class."""
    
    # Server settings
    server_name: str = "knowledge-base-mcp"
    server_version: str = "1.0.0"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Component configurations
    obsidian: ObsidianConfig = ObsidianConfig()
    pdf: PDFConfig = PDFConfig()
    knowledge_base: KnowledgeBaseConfig = KnowledgeBaseConfig()
    
    # Processing settings
    timezone: str = os.getenv("TIMEZONE", "UTC")
    max_concurrent_processes: int = int(os.getenv("MAX_CONCURRENT_PROCESSES", "3"))
    
    def validate_paths(self) -> List[str]:
        """Validate that required paths exist."""
        errors = []
        
        if not self.obsidian.vault_path or not Path(self.obsidian.vault_path).exists():
            errors.append(f"Obsidian vault path does not exist: {self.obsidian.vault_path}")
        
        for pdf_dir in self.pdf.pdf_directories:
            if not Path(pdf_dir).exists():
                errors.append(f"PDF directory does not exist: {pdf_dir}")
        
        return errors
    
    def create_missing_directories(self) -> List[str]:
        """Create missing directories and return list of created paths."""
        created = []
        
        # Create PDF directories
        for pdf_dir in self.pdf.pdf_directories:
            pdf_path = Path(pdf_dir)
            if not pdf_path.exists():
                pdf_path.mkdir(parents=True, exist_ok=True)
                created.append(str(pdf_path))
        
        # Create vector database directory
        vector_db_path = Path(self.pdf.vector_db_path)
        if not vector_db_path.exists():
            vector_db_path.mkdir(parents=True, exist_ok=True)
            created.append(str(vector_db_path))
        
        return created
    
    class Config:
        env_prefix = "KB_" 