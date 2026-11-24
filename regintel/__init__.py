"""RegIntel AI — Retrieval-Augmented Generation for U.S. banking regulatory research.

This package is organized as a classic RAG pipeline with pluggable components so
that heavy, optional dependencies (PyMuPDF, ChromaDB, Sentence Transformers, the
Anthropic SDK) are isolated behind small interfaces. The core logic — chunking,
retrieval ranking, citation validation and prompt construction — has no heavy
dependencies and is fully unit-tested.
"""

__version__ = "0.1.0"
