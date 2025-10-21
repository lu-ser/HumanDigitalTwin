"""
Ontology validation module for Schema.org-based triplet matching.
"""

from .ontology_loader import SchemaOrgLoader
from .embedding_service import EmbeddingService
from .triple_matcher import TripleMatcher

__all__ = [
    "SchemaOrgLoader",
    "EmbeddingService", 
    "TripleMatcher",
]
