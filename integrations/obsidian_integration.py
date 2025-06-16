"""
Obsidian Integration for Knowledge Base MCP Server
Handles reading from and writing to Obsidian vaults
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import re
import frontmatter

from config import Config
from models.knowledge_models import ObsidianNote, KnowledgeExtraction, SearchResult

logger = logging.getLogger(__name__)

class ObsidianIntegration:
    """Handles Obsidian vault operations for the knowledge base."""
    
    def __init__(self, config: Config):
        self.config = config
        self.vault_path = Path(self.config.obsidian.vault_path)
        self.notes_folder = self.config.obsidian.notes_folder
        self.templates_folder = self.config.obsidian.templates_folder
        
        # Validate vault path
        if not self.vault_path.exists():
            raise ValueError(f"Obsidian vault path does not exist: {self.vault_path}")
        
        logger.info(f"Initialized Obsidian integration for vault: {self.vault_path}")
    
    async def create_note_from_knowledge(
        self, 
        topic: str, 
        knowledge_extraction: KnowledgeExtraction,
        note_type: str = "general"
    ) -> ObsidianNote:
        """Create an Obsidian note from extracted knowledge."""
        try:
            # Generate note content based on type
            content = self._generate_note_content(knowledge_extraction, note_type)
            
            # Create filename (sanitize for filesystem)
            safe_title = self._sanitize_filename(f"{topic}")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{safe_title}_{timestamp}.md"
            
            # Determine file path
            if self.notes_folder:
                notes_dir = self.vault_path / self.notes_folder
                notes_dir.mkdir(exist_ok=True)
                file_path = notes_dir / filename
            else:
                file_path = self.vault_path / filename
            
            # Generate tags
            tags = self._generate_tags(knowledge_extraction)
            
            # Create ObsidianNote
            note = ObsidianNote(
                title=topic,
                content=content,
                file_path=str(file_path),
                tags=tags,
                created_date=datetime.now(),
                modified_date=datetime.now(),
                source_documents=knowledge_extraction.source_documents,
                topic=topic,
                kb_generated=True
            )
            
            # Write to file
            await self._write_note_to_file(note)
            
            logger.info(f"Created Obsidian note: {filename}")
            return note
            
        except Exception as e:
            logger.error(f"Error creating note from knowledge: {e}")
            raise
    
    def _generate_note_content(self, knowledge: KnowledgeExtraction, note_type: str) -> str:
        """Generate note content based on the knowledge extraction and note type."""
        
        if note_type == "summary":
            return self._generate_summary_note(knowledge)
        elif note_type == "detailed":
            return self._generate_detailed_note(knowledge)
        elif note_type == "flashcards":
            return self._generate_flashcard_note(knowledge)
        else:
            return self._generate_general_note(knowledge)
    
    def _generate_general_note(self, knowledge: KnowledgeExtraction) -> str:
        """Generate a general-purpose note."""
        content = f"# {knowledge.topic}\n\n"
        
        # Summary section
        content += "## Summary\n\n"
        content += f"{knowledge.summary}\n\n"
        
        # Key points
        if knowledge.key_points:
            content += "## Key Points\n\n"
            for point in knowledge.key_points:
                content += f"- {point}\n"
            content += "\n"
        
        # Related concepts
        if knowledge.related_concepts:
            content += "## Related Concepts\n\n"
            for concept in knowledge.related_concepts:
                # Create Obsidian links
                concept_link = f"[[{concept}]]"
                content += f"- {concept_link}\n"
            content += "\n"
        
        # Questions for further study
        if knowledge.questions:
            content += "## Study Questions\n\n"
            for question in knowledge.questions:
                content += f"- {question}\n"
            content += "\n"
        
        # Citations
        if knowledge.citations and self.config.knowledge_base.include_citations:
            content += "## Sources\n\n"
            for i, citation in enumerate(knowledge.citations, 1):
                source = citation.get('source', 'Unknown')
                page = citation.get('page', '')
                page_info = f" (p. {page})" if page else ""
                content += f"{i}. {source}{page_info}\n"
            content += "\n"
        
        return content
    
    def _generate_summary_note(self, knowledge: KnowledgeExtraction) -> str:
        """Generate a concise summary note."""
        content = f"# {knowledge.topic} - Summary\n\n"
        content += f"{knowledge.summary}\n\n"
        
        if knowledge.key_points:
            content += "## Key Takeaways\n\n"
            for point in knowledge.key_points[:5]:  # Limit to top 5
                content += f"- {point}\n"
            content += "\n"
        
        return content
    
    def _generate_detailed_note(self, knowledge: KnowledgeExtraction) -> str:
        """Generate a detailed comprehensive note."""
        content = f"# {knowledge.topic}\n\n"
        
        # Overview
        content += "## Overview\n\n"
        content += f"{knowledge.summary}\n\n"
        
        # Detailed key points
        if knowledge.key_points:
            content += "## Detailed Analysis\n\n"
            for i, point in enumerate(knowledge.key_points, 1):
                content += f"### {i}. {point}\n\n"
                content += "*(Add your notes and analysis here)*\n\n"
        
        # Related concepts with placeholders for expansion
        if knowledge.related_concepts:
            content += "## Connected Topics\n\n"
            for concept in knowledge.related_concepts:
                content += f"### [[{concept}]]\n\n"
                content += "*(Expand on the relationship between this concept and the main topic)*\n\n"
        
        # Study materials
        content += "## Study Materials\n\n"
        content += "### Questions to Explore\n\n"
        if knowledge.questions:
            for question in knowledge.questions:
                content += f"- [ ] {question}\n"
        content += "\n"
        
        content += "### Practice Problems\n\n"
        content += "*(Add practice problems or exercises here)*\n\n"
        
        return content
    
    def _generate_flashcard_note(self, knowledge: KnowledgeExtraction) -> str:
        """Generate a note optimized for flashcard creation."""
        content = f"# {knowledge.topic} - Flashcards\n\n"
        
        if knowledge.flashcards:
            content += "## Flashcards\n\n"
            for i, card in enumerate(knowledge.flashcards, 1):
                front = card.get('front', card.get('question', ''))
                back = card.get('back', card.get('answer', ''))
                content += f"### Card {i}\n\n"
                content += f"**Front:** {front}\n\n"
                content += f"**Back:** {back}\n\n"
                content += "---\n\n"
        
        # Generate additional flashcards from key points
        if knowledge.key_points:
            content += "## Generated from Key Points\n\n"
            for i, point in enumerate(knowledge.key_points, 1):
                content += f"### Card {i + len(knowledge.flashcards)}\n\n"
                content += f"**Front:** What is important about {knowledge.topic}?\n\n"
                content += f"**Back:** {point}\n\n"
                content += "---\n\n"
        
        return content
    
    def _generate_tags(self, knowledge: KnowledgeExtraction) -> List[str]:
        """Generate appropriate tags for the note."""
        tags = []
        
        # Add knowledge base prefix
        tags.append(f"{self.config.obsidian.tag_prefix}generated")
        
        # Add topic-based tag
        topic_tag = self._sanitize_tag(knowledge.topic)
        tags.append(f"{self.config.obsidian.tag_prefix}{topic_tag}")
        
        # Add tags based on related concepts
        for concept in knowledge.related_concepts[:3]:  # Limit to avoid too many tags
            concept_tag = self._sanitize_tag(concept)
            tags.append(f"{self.config.obsidian.tag_prefix}{concept_tag}")
        
        # Add source-based tags
        for source in knowledge.source_documents[:2]:  # Limit source tags
            source_name = Path(source).stem
            source_tag = self._sanitize_tag(source_name)
            tags.append(f"source/{source_tag}")
        
        return tags
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for filesystem compatibility."""
        # Replace problematic characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Remove excessive whitespace and special characters
        sanitized = re.sub(r'[^\w\s-]', '', sanitized).strip()
        # Replace spaces with underscores
        sanitized = re.sub(r'\s+', '_', sanitized)
        # Limit length
        return sanitized[:100]
    
    def _sanitize_tag(self, tag: str) -> str:
        """Sanitize tag for Obsidian compatibility."""
        # Convert to lowercase and replace spaces/special chars
        sanitized = re.sub(r'[^\w\s-]', '', tag.lower()).strip()
        sanitized = re.sub(r'\s+', '_', sanitized)
        return sanitized[:30]
    
    async def _write_note_to_file(self, note: ObsidianNote):
        """Write an ObsidianNote to file."""
        try:
            file_content = note.to_file_content()
            
            # Ensure directory exists
            Path(note.file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Write file
            with open(note.file_path, 'w', encoding='utf-8') as f:
                f.write(file_content)
                
        except Exception as e:
            logger.error(f"Error writing note to file {note.file_path}: {e}")
            raise
    
    async def read_note(self, file_path: str) -> Optional[ObsidianNote]:
        """Read an existing Obsidian note."""
        try:
            if not Path(file_path).exists():
                return None
            
            return ObsidianNote.from_file(file_path)
            
        except Exception as e:
            logger.error(f"Error reading note {file_path}: {e}")
            return None
    
    async def update_note(self, note: ObsidianNote, new_content: str = None, new_tags: List[str] = None):
        """Update an existing note."""
        try:
            if new_content:
                note.content = new_content
            
            if new_tags:
                note.tags = new_tags
            
            note.modified_date = datetime.now()
            
            await self._write_note_to_file(note)
            logger.info(f"Updated note: {Path(note.file_path).name}")
            
        except Exception as e:
            logger.error(f"Error updating note: {e}")
            raise
    
    async def search_notes(self, query: str) -> List[ObsidianNote]:
        """Search existing notes in the vault."""
        try:
            matching_notes = []
            
            # Search in the notes folder if specified, otherwise entire vault
            search_path = self.vault_path / self.notes_folder if self.notes_folder else self.vault_path
            
            # Find all markdown files
            for md_file in search_path.rglob("*.md"):
                try:
                    note = await self.read_note(str(md_file))
                    if note and self._note_matches_query(note, query):
                        matching_notes.append(note)
                except Exception as e:
                    logger.warning(f"Error reading note {md_file}: {e}")
            
            logger.info(f"Found {len(matching_notes)} notes matching query: {query}")
            return matching_notes
            
        except Exception as e:
            logger.error(f"Error searching notes: {e}")
            return []
    
    def _note_matches_query(self, note: ObsidianNote, query: str) -> bool:
        """Check if a note matches the search query."""
        query_lower = query.lower()
        
        # Check title
        if query_lower in note.title.lower():
            return True
        
        # Check content
        if query_lower in note.content.lower():
            return True
        
        # Check tags
        for tag in note.tags:
            if query_lower in tag.lower():
                return True
        
        # Check topic
        if note.topic and query_lower in note.topic.lower():
            return True
        
        return False
    
    async def get_vault_stats(self) -> Dict[str, Any]:
        """Get statistics about the Obsidian vault."""
        try:
            total_notes = 0
            kb_generated_notes = 0
            total_size = 0
            
            for md_file in self.vault_path.rglob("*.md"):
                total_notes += 1
                total_size += md_file.stat().st_size
                
                try:
                    note = await self.read_note(str(md_file))
                    if note and note.kb_generated:
                        kb_generated_notes += 1
                except:
                    pass
            
            stats = {
                "vault_path": str(self.vault_path),
                "total_notes": total_notes,
                "kb_generated_notes": kb_generated_notes,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "notes_folder": self.notes_folder,
                "templates_folder": self.templates_folder
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting vault stats: {e}")
            return {"error": str(e)}
    
    async def create_template_note(self, template_name: str, variables: Dict[str, str] = None) -> str:
        """Create a note based on a template."""
        try:
            template_path = self.vault_path / self.templates_folder / f"{template_name}.md"
            
            if not template_path.exists():
                logger.warning(f"Template not found: {template_path}")
                return ""
            
            # Read template
            with open(template_path, 'r', encoding='utf-8') as f:
                template_content = f.read()
            
            # Replace variables if provided
            if variables:
                for key, value in variables.items():
                    placeholder = f"{{{{{key}}}}}"
                    template_content = template_content.replace(placeholder, value)
            
            return template_content
            
        except Exception as e:
            logger.error(f"Error creating note from template {template_name}: {e}")
            return ""
    
    async def create_note_with_content(
        self,
        title: str,
        content: str,
        tags: List[str] = None,
        source_documents: List[str] = None,
        note_type: str = "general"
    ) -> ObsidianNote:
        """Create an Obsidian note with LLM-generated content."""
        try:
            # Create filename (sanitize for filesystem) 
            safe_title = self._sanitize_filename(title)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{safe_title}_{timestamp}.md"
            
            # Determine file path
            if self.notes_folder:
                notes_dir = self.vault_path / self.notes_folder
                notes_dir.mkdir(exist_ok=True)
                file_path = notes_dir / filename
            else:
                file_path = self.vault_path / filename
            
            # Process tags (add defaults if not provided)
            if not tags:
                tags = []
            
            # Add knowledge base tags
            kb_tags = [
                f"{self.config.obsidian.tag_prefix}generated",
                f"{self.config.obsidian.tag_prefix}llm_created",
                f"note_type/{note_type}"
            ]
            
            # Add topic-based tag from title
            topic_tag = self._sanitize_tag(title)
            kb_tags.append(f"{self.config.obsidian.tag_prefix}{topic_tag}")
            
            # Combine provided tags with generated ones
            all_tags = list(set(tags + kb_tags))
            
            # Process source documents
            if not source_documents:
                source_documents = []
            
            # Create ObsidianNote
            note = ObsidianNote(
                title=title,
                content=content,
                file_path=str(file_path),
                tags=all_tags,
                created_date=datetime.now(),
                modified_date=datetime.now(),
                source_documents=source_documents,
                topic=title,
                kb_generated=True
            )
            
            # Write to file
            await self._write_note_to_file(note)
            
            logger.info(f"Created Obsidian note with LLM content: {filename}")
            return note
            
        except Exception as e:
            logger.error(f"Error creating note with content: {e}")
            raise 