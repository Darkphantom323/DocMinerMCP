"""
Data models for Knowledge Base MCP Server
"""

from datetime import datetime, date
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from pydantic import BaseModel, Field
import frontmatter

class PDFDocument(BaseModel):
    """Represents a PDF document in the knowledge base."""
    
    file_path: str
    title: str
    author: Optional[str] = None
    creation_date: Optional[datetime] = None
    modification_date: Optional[datetime] = None
    page_count: int = 0
    file_size: int = 0
    
    # Extracted content
    full_text: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Processing info
    processed_at: Optional[datetime] = None
    embedding_generated: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        data = {
            "file_path": self.file_path,
            "title": self.title,
            "author": self.author,
            "creation_date": self.creation_date.isoformat() if self.creation_date else None,
            "modification_date": self.modification_date.isoformat() if self.modification_date else None,
            "page_count": self.page_count,
            "file_size": self.file_size,
            "metadata": self.metadata,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "embedding_generated": self.embedding_generated
        }
        # Filter out None values for database compatibility
        return {k: v for k, v in data.items() if v is not None}

class TextChunk(BaseModel):
    """Represents a chunk of text from a document."""
    
    source_file: str
    chunk_id: str
    content: str
    page_number: Optional[int] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    
    # Metadata
    chunk_type: str = "text"  # text, table, image_caption, etc.
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = {
            "source_file": self.source_file,
            "chunk_id": self.chunk_id,
            "content": self.content,
            "page_number": self.page_number,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "chunk_type": self.chunk_type
        }
        # Filter out None values for ChromaDB compatibility
        return {k: v for k, v in data.items() if v is not None}

class SearchResult(BaseModel):
    """Represents a search result from the knowledge base."""
    
    chunk: TextChunk
    similarity_score: float
    source_document: str
    context: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk": self.chunk.to_dict(),
            "similarity_score": self.similarity_score,
            "source_document": self.source_document,
            "context": self.context
        }

class ObsidianNote(BaseModel):
    """Represents an Obsidian note."""
    
    title: str
    content: str
    file_path: str
    
    # Frontmatter
    tags: List[str] = Field(default_factory=list)
    aliases: List[str] = Field(default_factory=list)
    created_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    
    # Knowledge base specific
    source_documents: List[str] = Field(default_factory=list)
    topic: Optional[str] = None
    kb_generated: bool = False
    
    @classmethod
    def from_file(cls, file_path: str) -> 'ObsidianNote':
        """Create an ObsidianNote from a file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            post = frontmatter.load(f)
        
        return cls(
            title=Path(file_path).stem,
            content=post.content,
            file_path=file_path,
            tags=post.metadata.get('tags', []),
            aliases=post.metadata.get('aliases', []),
            created_date=post.metadata.get('created_date'),
            modified_date=post.metadata.get('modified_date'),
            source_documents=post.metadata.get('source_documents', []),
            topic=post.metadata.get('topic'),
            kb_generated=post.metadata.get('kb_generated', False)
        )
    
    def to_file_content(self) -> str:
        """Convert the note to file content with frontmatter."""
        metadata = {
            'tags': self.tags,
            'aliases': self.aliases,
            'created_date': self.created_date.isoformat() if self.created_date else None,
            'modified_date': self.modified_date.isoformat() if self.modified_date else None,
            'source_documents': self.source_documents,
            'topic': self.topic,
            'kb_generated': self.kb_generated
        }
        
        # Remove None values
        metadata = {k: v for k, v in metadata.items() if v is not None}
        
        post = frontmatter.Post(self.content, **metadata)
        return frontmatter.dumps(post)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content,
            "file_path": self.file_path,
            "tags": self.tags,
            "aliases": self.aliases,
            "created_date": self.created_date.isoformat() if self.created_date else None,
            "modified_date": self.modified_date.isoformat() if self.modified_date else None,
            "source_documents": self.source_documents,
            "topic": self.topic,
            "kb_generated": self.kb_generated
        }

class KnowledgeExtraction(BaseModel):
    """Represents extracted knowledge from documents."""
    
    topic: str
    summary: str
    key_points: List[str] = Field(default_factory=list)
    related_concepts: List[str] = Field(default_factory=list)
    
    # Source information
    source_documents: List[str] = Field(default_factory=list)
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Generated content
    flashcards: List[Dict[str, str]] = Field(default_factory=list)
    questions: List[str] = Field(default_factory=list)
    
    # Metadata
    extraction_date: datetime = Field(default_factory=datetime.now)
    confidence_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "summary": self.summary,
            "key_points": self.key_points,
            "related_concepts": self.related_concepts,
            "source_documents": self.source_documents,
            "citations": self.citations,
            "flashcards": self.flashcards,
            "questions": self.questions,
            "extraction_date": self.extraction_date.isoformat(),
            "confidence_score": self.confidence_score
        }

class LearningProgress(BaseModel):
    """Tracks learning progress for topics."""
    
    topic: str
    notes_created: int = 0
    documents_reviewed: int = 0
    flashcards_studied: int = 0
    questions_answered: int = 0
    
    # Progress metrics
    understanding_level: int = 0  # 0-100
    last_reviewed: Optional[datetime] = None
    next_review: Optional[datetime] = None
    
    # Study session data
    study_sessions: List[Dict[str, Any]] = Field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "notes_created": self.notes_created,
            "documents_reviewed": self.documents_reviewed,
            "flashcards_studied": self.flashcards_studied,
            "questions_answered": self.questions_answered,
            "understanding_level": self.understanding_level,
            "last_reviewed": self.last_reviewed.isoformat() if self.last_reviewed else None,
            "next_review": self.next_review.isoformat() if self.next_review else None,
            "study_sessions": self.study_sessions
        }

class NoteGenerationRequest(BaseModel):
    """Request for generating a note on a specific topic."""
    
    topic: str
    focus_areas: List[str] = Field(default_factory=list)
    note_type: str = "general"  # general, summary, detailed, flashcards
    include_citations: bool = True
    max_length: Optional[int] = None
    
    # Search parameters
    search_query: Optional[str] = None
    max_sources: int = 10
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "focus_areas": self.focus_areas,
            "note_type": self.note_type,
            "include_citations": self.include_citations,
            "max_length": self.max_length,
            "search_query": self.search_query,
            "max_sources": self.max_sources
        } 