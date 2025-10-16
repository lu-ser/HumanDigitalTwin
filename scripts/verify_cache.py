"""
Script to verify and clean embeddings cache.
"""
import pickle
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ontology import SchemaOrgLoader

ontology = SchemaOrgLoader('data/ontology/schema.jsonld')
total_expected = len(ontology.get_all_classes()) + len(ontology.get_all_properties())

print(f"Expected: {total_expected}")

cache_file = Path('data/ontology/cache/embeddings_cache_cohere.pkl')
if cache_file.exists():
    with open(cache_file, 'rb') as f:
        cache = pickle.load(f)
    print(f"Cache entries: {len(cache)}")
    print(f"Difference: {len(cache) - total_expected}")
else:
    print("Cache not found")
