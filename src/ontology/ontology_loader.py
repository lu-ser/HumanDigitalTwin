"""
Loader for Schema.org ontology.
Parses the JSON-LD format and extracts classes, properties, and their descriptions.
"""

import json
from typing import Dict, List, Set
from pathlib import Path


class SchemaOrgLoader:
    def __init__(self, schema_path: str = "data/schema.jsonld"):
        self.schema_path = Path(schema_path)
        self.classes: Dict[str, Dict] = {}
        self.properties: Dict[str, Dict] = {}
        self.load_schema()

    def load_schema(self):
        """Load and parse Schema.org ontology from JSON-LD file."""
        with open(self.schema_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract graph entries
        graph = data.get('@graph', [])

        for entry in graph:
            entry_type = entry.get('@type')
            entry_id = entry.get('@id', '')

            # Skip internal Schema.org definitions
            if not entry_id.startswith('schema:'):
                continue

            name = entry_id.replace('schema:', '')

            # Process classes (Types)
            if 'rdfs:Class' in (entry_type if isinstance(entry_type, list) else [entry_type]):
                self.classes[name] = {
                    'id': entry_id,
                    'label': entry.get('rdfs:label', name),
                    'comment': entry.get('rdfs:comment', ''),
                    'subClassOf': self._extract_refs(entry.get('rdfs:subClassOf', [])),
                }

            # Process properties
            elif 'rdf:Property' in (entry_type if isinstance(entry_type, list) else [entry_type]):
                self.properties[name] = {
                    'id': entry_id,
                    'label': entry.get('rdfs:label', name),
                    'comment': entry.get('rdfs:comment', ''),
                    'domainIncludes': self._extract_refs(entry.get('schema:domainIncludes', [])),
                    'rangeIncludes': self._extract_refs(entry.get('schema:rangeIncludes', [])),
                }

    def _extract_refs(self, ref_data) -> List[str]:
        """Extract schema references from various formats."""
        if not ref_data:
            return []

        if isinstance(ref_data, dict):
            ref_id = ref_data.get('@id', '')
            return [ref_id.replace('schema:', '')] if ref_id.startswith('schema:') else []

        if isinstance(ref_data, list):
            refs = []
            for item in ref_data:
                if isinstance(item, dict):
                    ref_id = item.get('@id', '')
                    if ref_id.startswith('schema:'):
                        refs.append(ref_id.replace('schema:', ''))
            return refs

        return []

    def get_class_info(self, class_name: str) -> Dict:
        """Get information about a specific class."""
        return self.classes.get(class_name, {})

    def get_property_info(self, property_name: str) -> Dict:
        """Get information about a specific property."""
        return self.properties.get(property_name, {})

    def get_all_classes(self) -> List[str]:
        """Get list of all class names."""
        return list(self.classes.keys())

    def get_all_properties(self) -> List[str]:
        """Get list of all property names."""
        return list(self.properties.keys())

    def get_class_description(self, class_name: str) -> str:
        """Get human-readable description of a class."""
        class_info = self.get_class_info(class_name)
        if not class_info:
            return ""
        return f"{class_info.get('label', class_name)}: {class_info.get('comment', '')}"

    def get_property_description(self, property_name: str) -> str:
        """Get human-readable description of a property."""
        prop_info = self.get_property_info(property_name)
        if not prop_info:
            return ""
        return f"{prop_info.get('label', property_name)}: {prop_info.get('comment', '')}"
