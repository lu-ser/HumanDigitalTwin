# Ontology Validation Module

## Overview
This module provides Schema.org-based semantic validation for extracted RDF triplets using embedding similarity.

## Components

### Core Modules
- **ontology_loader.py**: Loads and parses Schema.org JSON-LD ontology
- **embedding_service.py**: Manages embeddings (Cohere/Mistral/Gemma/Qwen) with caching
- **embedding_cache.py**: Persistent cache for embeddings to avoid rate limits
- **triple_matcher.py**: Multi-branch matching algorithm for triplet validation
- **predicate_mappings.py**: Predicate hint mappings for better matching
- **schema_downloader.py**: Auto-downloads Schema.org ontology on first run

## Configuration
Add to `config.yaml`:
```yaml
ontology:
  schema_url: "https://schema.org/version/latest/schemaorg-current-https.jsonld"
  schema_path: "data/ontology/schema.jsonld"
  cache_dir: "data/ontology/cache"
  embedding_provider: "cohere"
  validation_threshold: 0.5
  rate_limit_delay: 0.1
```

## Workflow

1. **Extraction** (Tab 1):
   - Extract triplets from text using LangGraph
   - Triplets saved in `st.session_state["extracted_triplets"]`

2. **Validation** (Tab 2):
   - First run: Auto-downloads schema.jsonld + pre-computes embeddings (~2min with Cohere)
   - Subsequent runs: Loads from cache (instant)
   - Validates each triplet against Schema.org classes/properties
   - Shows results split by confidence threshold

## Matching Algorithm

Uses multi-branch exploration:
- **Branch 1**: Predicate-driven (match predicate → filter subject/object by domain/range)
- **Branch 2**: Subject-driven (match subject → find properties → match object)
- **Branch 3**: Object-driven (match object → find properties → match subject)

Returns best branch by average score (μ).

## Output Format

```json
{
  "subject": {
    "value": "John",
    "matched_class": "Person",
    "confidence": 0.89,
    "top_candidates": [["Person", 0.89], ["Agent", 0.76], ...]
  },
  "predicate": {
    "value": "works_at",
    "matched_property": "worksFor",
    "confidence": 0.78,
    "top_candidates": [["worksFor", 0.78], ["employedBy", 0.65], ...]
  },
  "object": {
    "value": "Google",
    "matched_class": "Organization",
    "confidence": 0.92,
    "top_candidates": [["Organization", 0.92], ["Corporation", 0.85], ...]
  },
  "mu": 0.863,
  "method_used": "predicate_driven",
  "branch_path": "Predicate(worksFor) → Subject(Person) + Object(Organization)"
}
```

## API Keys Required

- **Cohere**: `COHERE_API_KEY` in `.env`
- **Mistral**: `MISTRAL_API_KEY` in `.env`
- **Gemma/Qwen**: `HF_TOKEN` in `.env` (local models)

