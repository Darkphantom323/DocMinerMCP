"""
PDF Integration for Knowledge Base MCP Server
Handles PDF processing, text extraction, and vector search
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import hashlib
import pickle
from datetime import datetime

# PDF processing
import PyPDF2
import pdfplumber

# Vector database and embeddings
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings

# LangChain text splitters for advanced chunking
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    CharacterTextSplitter,
    MarkdownTextSplitter,
    PythonCodeTextSplitter
)
from langchain.docstore.document import Document as LangChainDocument

# PyMuPDF import with fallback
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

from config import Config
from models.knowledge_models import PDFDocument, TextChunk, SearchResult

logger = logging.getLogger(__name__)

class PDFIntegration:
    """Handles PDF processing and vector search for the knowledge base."""
    
    def __init__(self, config: Config):
        self.config = config
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        
        # Initialize LangChain text splitters
        self._initialize_text_splitters()
        
        # Initialize components
        self._initialize_embedding_model()
        self._initialize_vector_db()
    
    def _initialize_text_splitters(self):
        """Initialize LangChain text splitters for different content types."""
        try:
            # Primary splitter: Recursive Character Text Splitter (best for general text)
            self.recursive_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.pdf.chunk_size,
                chunk_overlap=self.config.pdf.chunk_overlap,
                length_function=len,
                separators=[
                    "\n\n",  # Double newlines (paragraphs)
                    "\n",    # Single newlines
                    ".",     # Sentences
                    "!",     # Exclamations
                    "?",     # Questions
                    ";",     # Semicolons
                    ",",     # Commas
                    " ",     # Spaces
                    ""       # Characters
                ]
            )
            
            # Token-based splitter for precise token control
            self.token_splitter = TokenTextSplitter(
                chunk_size=self.config.pdf.chunk_size // 4,  # Approximate tokens
                chunk_overlap=self.config.pdf.chunk_overlap // 4,
                encoding_name="cl100k_base"  # GPT-4 tokenizer
            )
            
            # Character splitter for simple splitting
            self.char_splitter = CharacterTextSplitter(
                chunk_size=self.config.pdf.chunk_size,
                chunk_overlap=self.config.pdf.chunk_overlap,
                separator="\n\n"
            )
            
            # Markdown splitter for structured content
            self.markdown_splitter = MarkdownTextSplitter(
                chunk_size=self.config.pdf.chunk_size,
                chunk_overlap=self.config.pdf.chunk_overlap
            )
            
            logger.info("Initialized LangChain text splitters")
            
        except Exception as e:
            logger.error(f"Failed to initialize text splitters: {e}")
            # Fallback to basic splitter
            self.recursive_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
    
    def _initialize_embedding_model(self):
        """Initialize the sentence transformer model."""
        try:
            # Try to initialize the embedding model
            self.embedding_model = SentenceTransformer(self.config.pdf.embedding_model)
            logger.info(f"Initialized embedding model: {self.config.pdf.embedding_model}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            # Don't raise, allow the system to work without embeddings
            self.embedding_model = None
            logger.warning("Embedding model disabled - search functionality will be limited")
    
    def _initialize_vector_db(self):
        """Initialize ChromaDB for vector storage."""
        try:
            # Create vector database directory if it doesn't exist
            Path(self.config.pdf.vector_db_path).mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client with simplified settings
            try:
                self.chroma_client = chromadb.PersistentClient(
                    path=self.config.pdf.vector_db_path,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True,
                        is_persistent=True
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to create PersistentClient with settings: {e}")
                # Fallback to simpler initialization
                self.chroma_client = chromadb.PersistentClient(
                    path=self.config.pdf.vector_db_path
                )
            
            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name="knowledge_base",
                metadata={"description": "Knowledge base document chunks"}
            )
            
            logger.info("Initialized ChromaDB vector database")
        except Exception as e:
            logger.error(f"Failed to initialize vector database: {e}")
            # Don't raise, allow the system to work without vector search
            self.chroma_client = None
            self.collection = None
            logger.warning("Vector database disabled - PDF processing will work but search will be limited")
    
    async def process_pdf_directory(self, directory_path: str) -> List[PDFDocument]:
        """Process all PDFs in a directory."""
        directory = Path(directory_path)
        if not directory.exists():
            logger.warning(f"Directory does not exist: {directory_path}")
            return []
        
        pdf_files = list(directory.glob("**/*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        processed_docs = []
        for pdf_file in pdf_files:
            try:
                doc = await self.process_pdf(str(pdf_file))
                if doc:
                    processed_docs.append(doc)
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {e}")
        
        return processed_docs
    
    async def process_pdf(self, file_path: str) -> Optional[PDFDocument]:
        """Process a single PDF file."""
        try:
            pdf_path = Path(file_path)
            if not pdf_path.exists():
                logger.error(f"PDF file does not exist: {file_path}")
                return None
            
            logger.info(f"Processing PDF: {pdf_path.name}")
            
            # Extract metadata and text
            metadata = self._extract_pdf_metadata(file_path)
            text_content = await self._extract_pdf_text(file_path)
            
            if not text_content.strip():
                logger.warning(f"No text content extracted from {pdf_path.name}")
                return None
            
            # Create PDFDocument
            doc = PDFDocument(
                file_path=file_path,
                title=metadata.get('title', pdf_path.stem),
                author=metadata.get('author'),
                creation_date=metadata.get('creation_date'),
                modification_date=metadata.get('modification_date'),
                page_count=metadata.get('page_count', 0),
                file_size=pdf_path.stat().st_size,
                full_text=text_content,
                metadata=metadata,
                processed_at=datetime.now()
            )
            
            # Generate advanced chunks using LangChain
            chunks = await self._create_advanced_text_chunks(doc)
            await self._store_chunks_in_vector_db(chunks)
            
            doc.embedding_generated = True
            logger.info(f"Successfully processed PDF: {pdf_path.name} ({len(chunks)} chunks)")
            
            return doc
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            return None
    
    def _extract_pdf_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from PDF."""
        metadata = {}
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Basic info
                metadata['page_count'] = len(pdf_reader.pages)
                
                # Document metadata
                if pdf_reader.metadata:
                    metadata['title'] = pdf_reader.metadata.get('/Title', '')
                    metadata['author'] = pdf_reader.metadata.get('/Author', '')
                    metadata['subject'] = pdf_reader.metadata.get('/Subject', '')
                    metadata['creator'] = pdf_reader.metadata.get('/Creator', '')
                    
                    # Dates
                    creation_date = pdf_reader.metadata.get('/CreationDate')
                    if creation_date:
                        try:
                            # Parse PDF date format
                            metadata['creation_date'] = datetime.strptime(
                                creation_date.replace('D:', '').split('+')[0].split('-')[0],
                                '%Y%m%d%H%M%S'
                            )
                        except:
                            pass
                    
                    mod_date = pdf_reader.metadata.get('/ModDate')
                    if mod_date:
                        try:
                            metadata['modification_date'] = datetime.strptime(
                                mod_date.replace('D:', '').split('+')[0].split('-')[0],
                                '%Y%m%d%H%M%S'
                            )
                        except:
                            pass
                            
        except Exception as e:
            logger.warning(f"Error extracting metadata from {file_path}: {e}")
        
        return metadata
    
    async def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text content from PDF using multiple methods."""
        text_content = ""
        
        # Method 1: Try pdfplumber (best for complex layouts)
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + "\n\n"
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed for {file_path}: {e}")
        
        # Method 2: Fallback to PyMuPDF if pdfplumber fails
        if not text_content.strip() and PYMUPDF_AVAILABLE:
            try:
                doc = fitz.open(file_path)
                for page_num in range(doc.page_count):
                    page = doc[page_num]
                    text_content += page.get_text() + "\n\n"
                doc.close()
            except Exception as e:
                logger.warning(f"PyMuPDF extraction failed for {file_path}: {e}")
        elif not text_content.strip() and not PYMUPDF_AVAILABLE:
            logger.warning(f"PyMuPDF not available for {file_path}, skipping method 2")
        
        # Method 3: Final fallback to PyPDF2
        if not text_content.strip():
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text_content += page.extract_text() + "\n\n"
            except Exception as e:
                logger.error(f"All PDF extraction methods failed for {file_path}: {e}")
        
        return text_content.strip()
    
    async def _create_advanced_text_chunks(self, doc: PDFDocument) -> List[TextChunk]:
        """Create advanced text chunks using LangChain text splitters."""
        chunks = []
        text = doc.full_text
        
        if not text.strip():
            logger.warning(f"No text content to chunk for {Path(doc.file_path).name}")
            return chunks
        
        try:
            # Create a LangChain document
            langchain_doc = LangChainDocument(
                page_content=text,
                metadata={
                    "source": doc.file_path,
                    "title": doc.title,
                    "author": doc.author,
                    "page_count": doc.page_count
                }
            )
            
            # Determine best splitting strategy based on content characteristics
            splitter = self._select_optimal_splitter(text)
            
            # Split the document
            split_docs = splitter.split_documents([langchain_doc])
            
            logger.info(f"LangChain splitter created {len(split_docs)} chunks for {Path(doc.file_path).name}")
            
            # Convert LangChain documents to TextChunk objects
            for i, split_doc in enumerate(split_docs):
                chunk_content = split_doc.page_content.strip()
                
                if not chunk_content or len(chunk_content) < 10:
                    continue
                
                # Calculate character positions (approximate)
                start_char = text.find(chunk_content[:50]) if len(chunk_content) >= 50 else text.find(chunk_content)
                if start_char == -1:
                    start_char = i * self.config.pdf.chunk_size
                end_char = start_char + len(chunk_content)
                
                # Determine chunk type based on content characteristics
                chunk_type = self._determine_chunk_type(chunk_content)
                
                chunk = TextChunk(
                    source_file=doc.file_path,
                    chunk_id=f"{Path(doc.file_path).stem}_langchain_{i}",
                    content=chunk_content,
                    start_char=start_char,
                    end_char=end_char,
                    chunk_type=chunk_type
                )
                
                chunks.append(chunk)
                
                # Add detailed metadata from LangChain document
                if hasattr(chunk, 'metadata'):
                    chunk.metadata = split_doc.metadata
            
            logger.info(f"Successfully created {len(chunks)} advanced chunks for {Path(doc.file_path).name}")
            
        except Exception as e:
            logger.error(f"Error creating advanced chunks: {e}")
            logger.info("Falling back to basic chunking method")
            
            # Fallback to basic chunking if LangChain fails
            chunks = await self._create_basic_text_chunks(doc)
        
        return chunks
    
    def _select_optimal_splitter(self, text: str):
        """Select the optimal text splitter based on content characteristics and configuration."""
        
        # If user specified a strategy, use it (unless it's auto)
        if self.config.pdf.chunking_strategy != "auto":
            strategy = self.config.pdf.chunking_strategy.lower()
            
            if strategy == "recursive":
                logger.info("Using RecursiveCharacterTextSplitter (user configured)")
                return self.recursive_splitter
            elif strategy == "token":
                logger.info("Using TokenTextSplitter (user configured)")
                return self.token_splitter
            elif strategy == "character":
                logger.info("Using CharacterTextSplitter (user configured)")
                return self.char_splitter
            elif strategy == "markdown":
                logger.info("Using MarkdownTextSplitter (user configured)")
                return self.markdown_splitter
        
        # Auto-selection based on content characteristics
        if self.config.pdf.detect_content_type:
            # Analyze text characteristics
            line_count = text.count('\n')
            paragraph_count = text.count('\n\n')
            has_markdown = any(marker in text for marker in ['#', '**', '##', '###', '- ', '* ', '1. '])
            has_code = any(marker in text for marker in ['```', 'def ', 'class ', 'import ', 'function'])
            
            # Decision logic for splitter selection
            if has_code and 'def ' in text:
                logger.info("Using PythonCodeTextSplitter for code-heavy content (auto-detected)")
                return PythonCodeTextSplitter(
                    chunk_size=self.config.pdf.chunk_size,
                    chunk_overlap=self.config.pdf.chunk_overlap
                )
            elif has_markdown:
                logger.info("Using MarkdownTextSplitter for structured content (auto-detected)")
                return self.markdown_splitter
            elif paragraph_count > line_count * 0.1 and self.config.pdf.split_on_paragraphs:
                logger.info("Using CharacterTextSplitter for paragraph-heavy content (auto-detected)")
                return self.char_splitter
            else:
                logger.info("Using RecursiveCharacterTextSplitter for general content (auto-detected)")
                return self.recursive_splitter
        else:
            # Default to recursive splitter if content detection is disabled
            logger.info("Using RecursiveCharacterTextSplitter (default, content detection disabled)")
            return self.recursive_splitter
    
    def _determine_chunk_type(self, content: str) -> str:
        """Determine the type of content in a chunk."""
        content_lower = content.lower()
        
        # Check for different content types
        if any(marker in content for marker in ['```', 'def ', 'class ', 'import ', 'function']):
            return "code"
        elif any(marker in content for marker in ['#', '##', '###']):
            return "heading"
        elif content.count('|') > 5 or 'table' in content_lower:
            return "table"
        elif any(marker in content for marker in ['figure', 'fig.', 'image', 'chart']):
            return "figure_reference"
        elif content.count('.') > len(content) / 50:  # High sentence density
            return "prose"
        elif any(marker in content for marker in ['- ', '* ', '1. ', '2. ']):
            return "list"
        else:
            return "text"

    async def _create_basic_text_chunks(self, doc: PDFDocument) -> List[TextChunk]:
        """Fallback basic chunking method (original implementation)."""
        chunks = []
        text = doc.full_text
        
        # Simple chunking strategy - split by size with overlap
        chunk_size = self.config.pdf.chunk_size
        overlap = self.config.pdf.chunk_overlap
        
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings near the chunk boundary
                for i in range(end, max(start + chunk_size - 200, start), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk_content = text[start:end].strip()
            
            if chunk_content:
                chunk = TextChunk(
                    source_file=doc.file_path,
                    chunk_id=f"{Path(doc.file_path).stem}_{chunk_id}",
                    content=chunk_content,
                    start_char=start,
                    end_char=end,
                    chunk_type="text"
                )
                chunks.append(chunk)
                chunk_id += 1
            
            # Move start position with overlap
            start = end - overlap
            if start >= end:
                break
        
        logger.info(f"Created {len(chunks)} basic chunks for {Path(doc.file_path).name}")
        return chunks
    
    async def _store_chunks_in_vector_db(self, chunks: List[TextChunk]):
        """Store text chunks in the vector database."""
        if not chunks:
            return
        
        if not self.embedding_model or not self.collection:
            logger.warning("Embedding model or vector database not available, skipping chunk storage")
            return
        
        try:
            # Filter out invalid chunks and prepare data for ChromaDB
            valid_chunks = []
            documents = []
            metadatas = []
            ids = []
            
            for chunk in chunks:
                # Skip empty chunks
                if not chunk.content or not chunk.content.strip():
                    continue
                
                # Skip chunks that are too short
                if len(chunk.content.strip()) < 10:
                    continue
                
                # Get metadata and ensure no None values
                metadata = chunk.to_dict()
                
                # Additional validation - ensure all required fields are present
                if not metadata.get('source_file') or not metadata.get('chunk_id'):
                    logger.warning(f"Skipping chunk with missing required fields: {chunk.chunk_id}")
                    continue
                
                valid_chunks.append(chunk)
                documents.append(chunk.content)
                metadatas.append(metadata)
                ids.append(chunk.chunk_id)
            
            if not valid_chunks:
                logger.warning("No valid chunks to store in vector database")
                return
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(documents)} valid chunks...")
            embeddings = self.embedding_model.encode(documents, show_progress_bar=True)
            
            # Store in ChromaDB
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings.tolist(),
                ids=ids
            )
            
            logger.info(f"Successfully stored {len(valid_chunks)} chunks in vector database")
            
        except Exception as e:
            logger.error(f"Error storing chunks in vector database: {e}")
            logger.warning("Continuing without vector storage")
    
    async def search_knowledge_base(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Enhanced search using multiple strategies for better results."""
        try:
            if not self.collection:
                logger.error("Vector database not initialized")
                return []
            
            if not self.embedding_model:
                logger.error("Embedding model not available")
                return []
            
            logger.info(f"Enhanced search for: '{query[:50]}...'")
            all_results = []
            
            # Strategy 1: Standard search with low threshold
            await self._low_threshold_search(query, all_results, max_results)
            
            # Strategy 2: Query variations
            await self._query_variations_search(query, all_results)
            
            # Strategy 3: Individual term search
            await self._individual_terms_search(query, all_results)
            
            # Strategy 4: Direct keyword search
            await self._direct_keyword_search(query, all_results)
            
            # Deduplicate and rank results
            unique_results = self._deduplicate_and_rank_results(all_results, query)
            
            # Limit results
            final_results = unique_results[:max_results]
            
            logger.info(f"Enhanced search found {len(final_results)} unique results for: {query[:50]}...")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in enhanced search: {e}")
            return []
    
    async def _low_threshold_search(self, query: str, all_results: List, max_results: int):
        """Search with very low similarity threshold."""
        try:
            # Use lower threshold for initial search
            low_threshold = min(0.05, self.config.pdf.similarity_threshold)
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=min(max_results * 2, 30)  # Get more candidates
            )
            
            if results['documents'] and results['documents'][0]:
                for document, metadata, distance in zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                ):
                    similarity_score = 1 - distance
                    
                    if similarity_score >= low_threshold:
                        # Create TextChunk from metadata
                        chunk = TextChunk(
                            source_file=metadata['source_file'],
                            chunk_id=metadata['chunk_id'],
                            content=document,
                            page_number=metadata.get('page_number'),
                            start_char=metadata.get('start_char'),
                            end_char=metadata.get('end_char'),
                            chunk_type=metadata.get('chunk_type', 'text')
                        )
                        
                        search_result = SearchResult(
                            chunk=chunk,
                            similarity_score=similarity_score,
                            source_document=Path(metadata['source_file']).name,
                            context="low_threshold_search"
                        )
                        
                        all_results.append(search_result)
            
            logger.debug(f"Low threshold search found {len([r for r in all_results if r.context == 'low_threshold_search'])} results")
            
        except Exception as e:
            logger.warning(f"Low threshold search failed: {e}")
    
    async def _query_variations_search(self, query: str, all_results: List):
        """Search with query variations."""
        try:
            variations = [
                f"What is {query}?",
                f"{query} definition",
                f"{query} explanation", 
                f"{query} method",
                f"{query} technique",
                f"about {query}"
            ]
            
            for variation in variations:
                try:
                    query_embedding = self.embedding_model.encode([variation])
                    results = self.collection.query(
                        query_embeddings=query_embedding.tolist(),
                        n_results=5
                    )
                    
                    if results['documents'] and results['documents'][0]:
                        for document, metadata, distance in zip(
                            results['documents'][0],
                            results['metadatas'][0],
                            results['distances'][0]
                        ):
                            similarity_score = (1 - distance) * 0.9  # Slight penalty for variations
                            
                            if similarity_score >= 0.05:  # Low threshold for variations
                                chunk = TextChunk(
                                    source_file=metadata['source_file'],
                                    chunk_id=metadata['chunk_id'],
                                    content=document,
                                    page_number=metadata.get('page_number'),
                                    start_char=metadata.get('start_char'),
                                    end_char=metadata.get('end_char'),
                                    chunk_type=metadata.get('chunk_type', 'text')
                                )
                                
                                search_result = SearchResult(
                                    chunk=chunk,
                                    similarity_score=similarity_score,
                                    source_document=Path(metadata['source_file']).name,
                                    context=f"variation: {variation}"
                                )
                                
                                all_results.append(search_result)
                
                except Exception:
                    continue
            
            variation_count = len([r for r in all_results if 'variation:' in r.context])
            logger.debug(f"Query variations found {variation_count} additional results")
            
        except Exception as e:
            logger.warning(f"Query variations search failed: {e}")
    
    async def _individual_terms_search(self, query: str, all_results: List):
        """Search for individual terms."""
        try:
            terms = query.split()
            
            if len(terms) > 1:
                for term in terms:
                    if len(term) > 3:  # Skip short terms
                        try:
                            query_embedding = self.embedding_model.encode([term])
                            results = self.collection.query(
                                query_embeddings=query_embedding.tolist(),
                                n_results=5
                            )
                            
                            if results['documents'] and results['documents'][0]:
                                for document, metadata, distance in zip(
                                    results['documents'][0],
                                    results['metadatas'][0],
                                    results['distances'][0]
                                ):
                                    similarity_score = (1 - distance) * 0.7  # Penalty for individual terms
                                    
                                    if similarity_score >= 0.05:
                                        chunk = TextChunk(
                                            source_file=metadata['source_file'],
                                            chunk_id=metadata['chunk_id'],
                                            content=document,
                                            page_number=metadata.get('page_number'),
                                            start_char=metadata.get('start_char'),
                                            end_char=metadata.get('end_char'),
                                            chunk_type=metadata.get('chunk_type', 'text')
                                        )
                                        
                                        search_result = SearchResult(
                                            chunk=chunk,
                                            similarity_score=similarity_score,
                                            source_document=Path(metadata['source_file']).name,
                                            context=f"term: {term}"
                                        )
                                        
                                        all_results.append(search_result)
                        
                        except Exception:
                            continue
            
            term_count = len([r for r in all_results if 'term:' in r.context])
            logger.debug(f"Individual terms search found {term_count} additional results")
            
        except Exception as e:
            logger.warning(f"Individual terms search failed: {e}")
    
    async def _direct_keyword_search(self, query: str, all_results: List):
        """Direct keyword search in document content."""
        try:
            # Get sample of documents for keyword search
            raw_data = self.collection.get(limit=2000)  # Sample limit
            
            if raw_data and 'documents' in raw_data:
                query_terms = query.lower().split()
                
                for i, doc_content in enumerate(raw_data['documents']):
                    doc_lower = doc_content.lower()
                    
                    # Check for term presence
                    matches = sum(1 for term in query_terms if term in doc_lower)
                    
                    if matches > 0:
                        metadata = raw_data.get('metadatas', [{}])[i] if i < len(raw_data.get('metadatas', [])) else {}
                        
                        chunk = TextChunk(
                            source_file=metadata.get('source_file', 'unknown'),
                            chunk_id=metadata.get('chunk_id', 'unknown'),
                            content=doc_content,
                            page_number=metadata.get('page_number'),
                            start_char=metadata.get('start_char'),
                            end_char=metadata.get('end_char'),
                            chunk_type=metadata.get('chunk_type', 'text')
                        )
                        
                        # Calculate keyword-based score
                        keyword_score = matches / len(query_terms)
                        
                        search_result = SearchResult(
                            chunk=chunk,
                            similarity_score=keyword_score,
                            source_document=Path(metadata.get('source_file', 'unknown')).name,
                            context="direct_keyword"
                        )
                        
                        all_results.append(search_result)
            
            keyword_count = len([r for r in all_results if r.context == 'direct_keyword'])
            logger.debug(f"Direct keyword search found {keyword_count} additional results")
            
        except Exception as e:
            logger.warning(f"Direct keyword search failed: {e}")
    
    def _deduplicate_and_rank_results(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """Remove duplicates and rank results by relevance."""
        
        # Deduplicate by chunk_id
        seen_chunks = set()
        unique_results = []
        
        for result in results:
            chunk_id = result.chunk.chunk_id
            if chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                unique_results.append(result)
        
        # Enhanced ranking
        query_terms = set(query.lower().split())
        
        for result in unique_results:
            content_lower = result.chunk.content.lower()
            
            # Keyword matching bonus
            keyword_matches = sum(1 for term in query_terms if term in content_lower)
            keyword_bonus = keyword_matches / len(query_terms) if query_terms else 0
            
            # Search strategy bonus
            strategy_bonus = 0.1 if 'low_threshold' in result.context else 0.05
            
            # Length preference (prefer substantial chunks)
            length_score = min(len(result.chunk.content) / 1000, 1.0)
            
            # Final score calculation
            result.similarity_score = (
                result.similarity_score * 0.6 +  # Original similarity
                keyword_bonus * 0.25 +          # Keyword matching
                strategy_bonus +                # Strategy preference  
                length_score * 0.05             # Length factor
            )
        
        # Sort by final score
        unique_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return unique_results
    
    async def get_document_summary(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get a summary of a specific document."""
        try:
            # Get document chunks
            chunks = []
            results = self.collection.get(
                where={"source_file": file_path}
            )
            
            if results['documents']:
                summary = {
                    "file_path": file_path,
                    "title": Path(file_path).stem,
                    "chunk_count": len(results['documents']),
                    "total_length": sum(len(doc) for doc in results['documents']),
                    "first_chunk_preview": results['documents'][0][:200] + "..." if results['documents'] else ""
                }
                return summary
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting document summary for {file_path}: {e}")
            return None
    
    async def rebuild_vector_database(self):
        """Rebuild the entire vector database from scratch."""
        try:
            logger.info("Rebuilding vector database...")
            
            # Clear existing collection
            self.chroma_client.delete_collection("knowledge_base")
            self.collection = self.chroma_client.create_collection(
                name="knowledge_base",
                metadata={"description": "Knowledge base document chunks"}
            )
            
            # Process all PDF directories
            all_docs = []
            for pdf_dir in self.config.pdf.pdf_directories:
                docs = await self.process_pdf_directory(pdf_dir)
                all_docs.extend(docs)
            
            logger.info(f"Vector database rebuild complete. Processed {len(all_docs)} documents.")
            return len(all_docs)
            
        except Exception as e:
            logger.error(f"Error rebuilding vector database: {e}")
            raise
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database."""
        try:
            if not self.collection:
                return {"error": "Vector database not initialized"}
            
            count = self.collection.count()
            
            # Get sample of documents to analyze
            sample_results = self.collection.get(limit=min(100, count))
            
            # Calculate statistics
            sources = set()
            total_length = 0
            chunk_types = {}
            
            if sample_results['metadatas']:
                for metadata in sample_results['metadatas']:
                    sources.add(metadata.get('source_file', 'unknown'))
                    chunk_type = metadata.get('chunk_type', 'unknown')
                    chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
                
                if sample_results['documents']:
                    total_length = sum(len(doc) for doc in sample_results['documents'])
            
            stats = {
                "total_chunks": count,
                "unique_documents": len(sources),
                "sample_size": len(sample_results['documents']) if sample_results['documents'] else 0,
                "average_chunk_length": total_length // len(sample_results['documents']) if sample_results['documents'] else 0,
                "chunk_types": chunk_types,
                "embedding_model": self.config.pdf.embedding_model,
                "database_path": self.config.pdf.vector_db_path,
                "chunking_method": "LangChain Advanced"
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {"error": str(e)} 