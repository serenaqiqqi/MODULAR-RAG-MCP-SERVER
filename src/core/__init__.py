"""
Core Layer - Core business logic.

This package contains the core business logic including:
- Configuration management (settings.py)
- Core data types (types.py) - shared contracts for all pipeline stages
- Query engine
- Response building
- Trace collection
"""

from src.core.types import Document, Chunk, ChunkRecord, Metadata, Vector, SparseVector

__all__ = [
    "Document",
    "Chunk", 
    "ChunkRecord",
    "Metadata",
    "Vector",
    "SparseVector"
]
#当别人写 from src.core import * 时，只导出这些名字。