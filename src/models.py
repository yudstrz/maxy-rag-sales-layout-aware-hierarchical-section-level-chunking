from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ==========================================
# Section Chunk Data Structure
# ==========================================
@dataclass
class SectionChunk:
    section_id: str
    section_path: str
    title: str
    full_text: str
    abstract: str
    source_type: str
    position: int
    metadata: dict = field(default_factory=dict)
    
    def to_langchain_doc(self, use_abstract: bool = False):
        from langchain_core.documents import Document
        content = self.abstract if use_abstract else self.full_text
        return Document(
            page_content=content,
            metadata={
                'section_id': self.section_id,
                'section_path': self.section_path,
                'title': self.title,
                'source_type': self.source_type,
                'position': self.position,
                **self.metadata
            }
        )
