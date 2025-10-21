"""
Triple matcher that assigns Schema.org classes to triple components.
Uses semantic matching via embeddings.
"""

from typing import Dict, List, Tuple
import numpy as np
from .ontology_loader import SchemaOrgLoader
from .embedding_service import EmbeddingService
from .predicate_mappings import get_predicate_hints


class TripleMatcher:
    def __init__(self, ontology_loader: SchemaOrgLoader, embedding_service: EmbeddingService, rate_limit_delay: float = 0.0):
        self.ontology = ontology_loader
        self.embeddings = embedding_service
        self.rate_limit_delay = rate_limit_delay

        # Cache for class embeddings
        self._class_embeddings_cache: Dict[str, List[float]] = {}
        self._property_embeddings_cache: Dict[str, List[float]] = {}

    def match_triple(self, subject_value: str, predicate_value: str, obj_value: str,
                     subject_type: str = None, predicate_type: str = None, obj_type: str = None) -> Dict:
        """
        Match a typed triple (matrice 2x3) using multi-branch exploration.

        Explores all possible combinations:
        - Branch 1: Predicate → Subject/Object (top N predicates)
        - Branch 2: Subject → Predicate → Object (top N subjects)
        - Branch 3: Object → Predicate → Subject (top N objects)

        Returns the branch with highest average similarity score, enriched with Schema.org types.

        Args:
            subject_value: Value of subject (e.g., "Marco")
            predicate_value: Value of predicate (e.g., "has_friend")
            obj_value: Value of object (e.g., "Luca")
            subject_type: LLM-extracted type (e.g., "Person") - will be refined with Schema.org
            predicate_type: LLM-extracted type (e.g., "Relationship")
            obj_type: LLM-extracted type (e.g., "Person")

        Returns:
            Dictionary with:
            - original types (from LLM extraction)
            - matched Schema.org classes (from semantic matching)
            - confidence scores
            - all explored branches
        """
        subject = subject_value
        predicate = predicate_value
        obj = obj_value
        all_branches = []

        # BRANCH 1: Predicate → Subject/Object (usa tipi LLM per matching semantico)
        predicate_branches = self._explore_predicate_branches(
            subject, predicate, obj,
            subject_type, predicate_type, obj_type,
            top_n=3
        )
        all_branches.extend(predicate_branches)

        # BRANCH 2: Subject → Predicate → Object (usa tipi LLM per matching semantico)
        subject_branches = self._explore_subject_branches(
            subject, predicate, obj,
            subject_type, predicate_type, obj_type,
            top_n=3
        )
        all_branches.extend(subject_branches)

        # BRANCH 3: Object → Predicate → Subject (usa tipi LLM per matching semantico)
        object_branches = self._explore_object_branches(
            subject, predicate, obj,
            subject_type, predicate_type, obj_type,
            top_n=3
        )
        all_branches.extend(object_branches)

        # Sort branches by mu (average score)
        all_branches.sort(key=lambda b: b['mu'], reverse=True)

        # Return best branch
        best_branch = all_branches[0] if all_branches else self._create_empty_result(subject, predicate, obj)

        # Enrich with original LLM types (riga 1 della matrice 2x3)
        best_branch['subject']['original_type'] = subject_type
        best_branch['predicate']['original_type'] = predicate_type
        best_branch['object']['original_type'] = obj_type

        # Add all branches for visualization (without circular references)
        # Create simplified branches without 'all_branches' field
        simplified_branches = []
        for branch in all_branches[:20]:
            simplified = {
                'subject': branch['subject'],
                'predicate': branch['predicate'],
                'object': branch['object'],
                'mu': branch['mu'],
                'method_used': branch['method_used'],
                'branch_path': branch['branch_path']
            }
            simplified_branches.append(simplified)

        best_branch['all_branches'] = simplified_branches

        return best_branch

    def _explore_predicate_branches(self, subject: str, predicate: str, obj: str,
                                    subject_type: str = None, predicate_type: str = None, obj_type: str = None,
                                    top_n: int = 5) -> List[Dict]:
        """Explore branches starting from predicate candidates."""
        branches = []

        # Get top N predicate candidates (usando il TIPO del predicate, non il valore)
        predicate_text = predicate_type if predicate_type else predicate
        predicate_candidates = self._match_to_property(predicate_text)[:top_n]  # No context!

        for pred_name, pred_score in predicate_candidates:
            # Get domain/range constraints
            prop_info = self.ontology.get_property_info(pred_name)
            domain_classes = prop_info.get('domainIncludes', [])
            range_classes = prop_info.get('rangeIncludes', [])

            if not domain_classes or not range_classes:
                continue

            # Match subject against domain classes (usando il TIPO, non il valore)
            subject_text = subject_type if subject_type else subject
            subject_candidates = self._match_to_class(subject_text)  # No context!
            subject_filtered = [(cls, score) for cls, score in subject_candidates if self._is_class_compatible(cls, domain_classes)]

            if not subject_filtered:
                continue

            # Match object against range classes (usando il TIPO, non il valore)
            obj_text = obj_type if obj_type else obj
            object_candidates = self._match_to_class(obj_text)  # No context!
            object_filtered = [(cls, score) for cls, score in object_candidates if self._is_class_compatible(cls, range_classes)]

            if not object_filtered:
                continue

            # Create branch
            subj_class, subj_score = subject_filtered[0]
            obj_class, obj_score = object_filtered[0]

            branch = {
                'subject': {
                    'value': subject,
                    'matched_class': subj_class,
                    'confidence': subj_score,
                    'top_candidates': subject_filtered[:10]
                },
                'predicate': {
                    'value': predicate,
                    'matched_property': pred_name,
                    'confidence': pred_score,
                    'top_candidates': predicate_candidates[:5]
                },
                'object': {
                    'value': obj,
                    'matched_class': obj_class,
                    'confidence': obj_score,
                    'top_candidates': object_filtered[:10]
                },
                'mu': (subj_score + pred_score + obj_score) / 3,
                'method_used': 'predicate_driven',
                'branch_path': f"Predicate({pred_name}) → Subject({subj_class}) + Object({obj_class})"
            }

            branches.append(branch)

        return branches

    def _explore_subject_branches(self, subject: str, predicate: str, obj: str,
                                  subject_type: str = None, predicate_type: str = None, obj_type: str = None,
                                  top_n: int = 5) -> List[Dict]:
        """Explore branches starting from subject candidates."""
        branches = []

        # Get top N subject candidates (usando il TIPO, non il valore)
        subject_text = subject_type if subject_type else subject
        subject_candidates = self._match_to_class(subject_text)[:top_n]  # No context!

        for subj_class, subj_score in subject_candidates:
            # Find properties with this class in domain
            properties_for_subject = self._find_properties_by_domain(subj_class)

            if not properties_for_subject:
                continue

            # Match predicate against these properties (usando il TIPO, non il valore)
            predicate_text = predicate_type if predicate_type else predicate
            predicate_filtered = self._match_to_property_filtered(predicate_text, properties_for_subject)

            if not predicate_filtered:
                continue

            pred_name, pred_score = predicate_filtered[0]

            # Get range constraints
            prop_info = self.ontology.get_property_info(pred_name)
            range_classes = prop_info.get('rangeIncludes', [])

            if not range_classes:
                continue

            # Match object against range classes (usando il TIPO, non il valore)
            obj_text = obj_type if obj_type else obj
            object_candidates = self._match_to_class(obj_text)  # No context!
            object_filtered = [(cls, score) for cls, score in object_candidates if self._is_class_compatible(cls, range_classes)]

            if not object_filtered:
                continue

            obj_class, obj_score = object_filtered[0]

            branch = {
                'subject': {
                    'value': subject,
                    'matched_class': subj_class,
                    'confidence': subj_score,
                    'top_candidates': subject_candidates[:10]
                },
                'predicate': {
                    'value': predicate,
                    'matched_property': pred_name,
                    'confidence': pred_score,
                    'top_candidates': predicate_filtered[:5]
                },
                'object': {
                    'value': obj,
                    'matched_class': obj_class,
                    'confidence': obj_score,
                    'top_candidates': object_filtered[:10]
                },
                'mu': (subj_score + pred_score + obj_score) / 3,
                'method_used': 'subject_driven',
                'branch_path': f"Subject({subj_class}) → Predicate({pred_name}) → Object({obj_class})"
            }

            branches.append(branch)

        return branches

    def _explore_object_branches(self, subject: str, predicate: str, obj: str,
                                 subject_type: str = None, predicate_type: str = None, obj_type: str = None,
                                 top_n: int = 5) -> List[Dict]:
        """Explore branches starting from object candidates."""
        branches = []

        # Get top N object candidates (usando il TIPO, non il valore)
        obj_text = obj_type if obj_type else obj
        object_candidates = self._match_to_class(obj_text)[:top_n]  # No context!

        for obj_class, obj_score in object_candidates:
            # Find properties with this class in range
            properties_for_object = self._find_properties_by_range(obj_class)

            if not properties_for_object:
                continue

            # Match predicate against these properties (usando il TIPO, non il valore)
            predicate_text = predicate_type if predicate_type else predicate
            predicate_filtered = self._match_to_property_filtered(predicate_text, properties_for_object)

            if not predicate_filtered:
                continue

            pred_name, pred_score = predicate_filtered[0]

            # Get domain constraints
            prop_info = self.ontology.get_property_info(pred_name)
            domain_classes = prop_info.get('domainIncludes', [])

            if not domain_classes:
                continue

            # Match subject against domain classes (usando il TIPO, non il valore)
            subject_text = subject_type if subject_type else subject
            subject_candidates = self._match_to_class(subject_text)  # No context!
            subject_filtered = [(cls, score) for cls, score in subject_candidates if self._is_class_compatible(cls, domain_classes)]

            if not subject_filtered:
                continue

            subj_class, subj_score = subject_filtered[0]

            branch = {
                'subject': {
                    'value': subject,
                    'matched_class': subj_class,
                    'confidence': subj_score,
                    'top_candidates': subject_filtered[:10]
                },
                'predicate': {
                    'value': predicate,
                    'matched_property': pred_name,
                    'confidence': pred_score,
                    'top_candidates': predicate_filtered[:5]
                },
                'object': {
                    'value': obj,
                    'matched_class': obj_class,
                    'confidence': obj_score,
                    'top_candidates': object_candidates[:10]
                },
                'mu': (subj_score + pred_score + obj_score) / 3,
                'method_used': 'object_driven',
                'branch_path': f"Object({obj_class}) → Predicate({pred_name}) → Subject({subj_class})"
            }

            branches.append(branch)

        return branches

    def _is_class_compatible(self, class_name: str, allowed_classes: List[str]) -> bool:
        """Check if a class is compatible with allowed classes (including parent classes)."""
        if class_name in allowed_classes:
            return True

        # Check parent classes
        class_info = self.ontology.get_class_info(class_name)
        parent_classes = class_info.get('subClassOf', [])

        return any(p in allowed_classes for p in parent_classes)

    def _find_properties_by_domain(self, class_name: str) -> List[str]:
        """Find properties that have class_name in their domain."""
        properties = []

        for prop_name in self.ontology.get_all_properties():
            prop_info = self.ontology.get_property_info(prop_name)
            domain_classes = prop_info.get('domainIncludes', [])

            if self._is_class_compatible(class_name, domain_classes):
                properties.append(prop_name)

        return properties

    def _find_properties_by_range(self, class_name: str) -> List[str]:
        """Find properties that have class_name in their range."""
        properties = []

        for prop_name in self.ontology.get_all_properties():
            prop_info = self.ontology.get_property_info(prop_name)
            range_classes = prop_info.get('rangeIncludes', [])

            if self._is_class_compatible(class_name, range_classes):
                properties.append(prop_name)

        return properties

    def _create_empty_result(self, subject: str, predicate: str, obj: str) -> Dict:
        """Create empty result when no branches found."""
        return {
            'subject': {'value': subject, 'matched_class': None, 'confidence': 0.0, 'top_candidates': []},
            'predicate': {'value': predicate, 'matched_property': None, 'confidence': 0.0, 'top_candidates': []},
            'object': {'value': obj, 'matched_class': None, 'confidence': 0.0, 'top_candidates': []},
            'mu': 0.0,
            'method_used': 'none',
            'branch_path': 'No valid branches found',
            'all_branches': []
        }

    def _match_predicate_driven(self, subject: str, predicate: str, obj: str) -> Dict:
        """Method 1: Match predicate first, then use domain/range to guide subject/object."""
        result = {
            'subject': {
                'value': subject,
                'matched_class': None,
                'confidence': 0.0,
                'top_candidates': []
            },
            'predicate': {
                'value': predicate,
                'matched_property': None,
                'confidence': 0.0,
                'top_candidates': []
            },
            'object': {
                'value': obj,
                'matched_class': None,
                'confidence': 0.0,
                'top_candidates': []
            }
        }

        # Match subject to a Schema.org class
        subject_match = self._match_to_class(subject, context=f"is related via '{predicate}'")
        result['subject']['matched_class'] = subject_match[0][0] if subject_match else None
        result['subject']['confidence'] = subject_match[0][1] if subject_match else 0.0
        result['subject']['top_candidates'] = subject_match[:10]

        # Match predicate to a Schema.org property
        predicate_match = self._match_to_property(predicate, context=f"connects {subject} to {obj}")
        result['predicate']['matched_property'] = predicate_match[0][0] if predicate_match else None
        result['predicate']['confidence'] = predicate_match[0][1] if predicate_match else 0.0
        result['predicate']['top_candidates'] = predicate_match[:5]

        # Match object to a Schema.org class
        object_match = self._match_to_class(obj, context=f"is related via '{predicate}'")
        result['object']['matched_class'] = object_match[0][0] if object_match else None
        result['object']['confidence'] = object_match[0][1] if object_match else 0.0
        result['object']['top_candidates'] = object_match[:10]

        # Refine matches based on property domain/range constraints
        result = self._refine_with_constraints(result)

        return result

    def _match_entity_driven(self, subject: str, predicate: str, obj: str) -> Dict:
        """Method 2: Match subject/object first, then find coherent predicates."""
        # Match subject and object independently
        subject_match = self._match_to_class(subject, context="")
        object_match = self._match_to_class(obj, context="")

        if not subject_match or not object_match:
            return None

        subject_class = subject_match[0][0]
        subject_score = subject_match[0][1]
        object_class = object_match[0][0]
        object_score = object_match[0][1]

        # Find properties that have subject_class in domain AND object_class in range
        coherent_properties = self._find_coherent_properties(subject_class, object_class)

        if not coherent_properties:
            return None  # No coherent predicates found, method 2 not applicable

        # Match predicate against only the coherent properties
        predicate_match = self._match_to_property_filtered(predicate, coherent_properties)

        if not predicate_match:
            return None

        result = {
            'subject': {
                'value': subject,
                'matched_class': subject_class,
                'confidence': subject_score,
                'top_candidates': subject_match[:10]
            },
            'predicate': {
                'value': predicate,
                'matched_property': predicate_match[0][0],
                'confidence': predicate_match[0][1],
                'top_candidates': predicate_match[:5]
            },
            'object': {
                'value': obj,
                'matched_class': object_class,
                'confidence': object_score,
                'top_candidates': object_match[:10]
            }
        }

        return result

    def _find_coherent_properties(self, subject_class: str, object_class: str) -> List[str]:
        """Find properties where subject_class is in domain AND object_class is in range."""
        coherent = []

        for prop_name in self.ontology.get_all_properties():
            prop_info = self.ontology.get_property_info(prop_name)
            domain_classes = prop_info.get('domainIncludes', [])
            range_classes = prop_info.get('rangeIncludes', [])

            # Check if subject_class is in domain (or parent class)
            subject_match = subject_class in domain_classes
            if not subject_match:
                # Check parent classes
                subject_info = self.ontology.get_class_info(subject_class)
                subject_parents = subject_info.get('subClassOf', [])
                subject_match = any(p in domain_classes for p in subject_parents)

            # Check if object_class is in range (or parent class)
            object_match = object_class in range_classes
            if not object_match:
                # Check parent classes
                object_info = self.ontology.get_class_info(object_class)
                object_parents = object_info.get('subClassOf', [])
                object_match = any(p in range_classes for p in object_parents)

            if subject_match and object_match:
                coherent.append(prop_name)

        return coherent

    def _match_to_property_filtered(self, text: str, allowed_properties: List[str]) -> List[Tuple[str, float]]:
        """Match predicate against only allowed properties."""
        # Get embedding for query (cached per evitare rate limit)
        query_text = f"{text}"
        query_embedding = self.embeddings.embed_text(
            query_text,
            input_type="search_document",  # Cached!
            rate_limit_delay=self.rate_limit_delay
        )

        similarities = []
        for property_name in allowed_properties:
            property_embedding = self._get_property_embedding(property_name)
            similarity = self.embeddings.cosine_similarity(query_embedding, property_embedding)
            similarities.append((property_name, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities

    def _calculate_average_score(self, result: Dict) -> float:
        """Calculate average similarity score across subject, predicate, object."""
        scores = [
            result['subject']['confidence'],
            result['predicate']['confidence'],
            result['object']['confidence']
        ]
        return sum(scores) / len(scores) if scores else 0.0

    def _match_to_class(self, text: str, context: str = "") -> List[Tuple[str, float]]:
        """
        Match text to Schema.org classes using semantic similarity.

        Args:
            text: Text to match
            context: Additional context for better matching

        Returns:
            List of (class_name, similarity_score) tuples, sorted by score
        """
        # Generate embedding for the input text with context
        # IMPORTANTE: usa "search_document" invece di "search_query" per cachare i tipi LLM
        # (Person, Place, etc. vengono riusati molte volte, quindi devono essere cachati)
        query_text = f"{text}. Context: {context}" if context else text
        query_embedding = self.embeddings.embed_text(
            query_text,
            input_type="search_document",  # Cached! Evita rate limit
            rate_limit_delay=self.rate_limit_delay
        )

        # Get embeddings for all classes
        # Se stiamo matchando tipi LLM (senza context), usa simple name matching
        use_simple = (context == "")  # No context = stiamo usando tipi LLM puri

        class_names = self.ontology.get_all_classes()
        similarities = []

        for class_name in class_names:
            class_embedding = self._get_class_embedding(class_name, use_simple_name=use_simple)
            similarity = self.embeddings.cosine_similarity(query_embedding, class_embedding)
            similarities.append((class_name, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities

    def _match_to_property(self, text: str, context: str = "") -> List[Tuple[str, float]]:
        """
        Match text to Schema.org properties using semantic similarity
        with predicate mapping hints.

        Args:
            text: Text to match
            context: Additional context for better matching

        Returns:
            List of (property_name, similarity_score) tuples, sorted by score
        """
        # Check for predicate mapping hints
        hints = get_predicate_hints(text)

        # Generate embedding for the input text with context
        # IMPORTANTE: usa "search_document" per cachare i tipi di predicato (Relationship)
        query_text = f"{text}. Context: {context}" if context else text
        query_embedding = self.embeddings.embed_text(
            query_text,
            input_type="search_document",  # Cached! Evita rate limit
            rate_limit_delay=self.rate_limit_delay
        )

        # Get embeddings for all properties
        property_names = self.ontology.get_all_properties()
        similarities = []

        for property_name in property_names:
            property_embedding = self._get_property_embedding(property_name)
            similarity = self.embeddings.cosine_similarity(query_embedding, property_embedding)

            # Boost if in hint list
            if hints and property_name in hints:
                similarity = min(1.0, similarity * 1.8)  # Strong boost for mapped predicates

            similarities.append((property_name, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities

    def _get_class_embedding(self, class_name: str, use_simple_name: bool = False) -> List[float]:
        """
        Get or compute embedding for a class.

        Args:
            class_name: Nome della classe Schema.org
            use_simple_name: Se True, usa solo il nome (per matching con tipi LLM puri)
        """
        # Cache separata per simple names
        cache_key = f"{class_name}_simple" if use_simple_name else class_name

        if cache_key not in self._class_embeddings_cache:
            if use_simple_name:
                # Per tipi LLM puri (Person, Place, Thing): usa SOLO il nome
                text = class_name
            else:
                # Per matching complesso: usa descrizione arricchita
                class_desc = self.ontology.get_class_description(class_name)
                class_info = self.ontology.get_class_info(class_name)

                parts = [f"{class_name}"]
                if class_desc:
                    parts.append(class_desc)

                parent_classes = class_info.get('subClassOf', [])
                if parent_classes:
                    parents_str = ", ".join(parent_classes[:3])
                    parts.append(f"Type of: {parents_str}")

                text = ". ".join(parts)

            embedding = self.embeddings.embed_text(
                text,
                input_type="search_document",
                rate_limit_delay=self.rate_limit_delay
            )
            self._class_embeddings_cache[cache_key] = embedding

        return self._class_embeddings_cache[cache_key]

    def _get_property_embedding(self, property_name: str) -> List[float]:
        """Get or compute embedding for a property with enriched context."""
        if property_name not in self._property_embeddings_cache:
            property_desc = self.ontology.get_property_description(property_name)
            property_info = self.ontology.get_property_info(property_name)

            # Build enriched description
            parts = [f"{property_name}"]

            # Add base description
            if property_desc:
                parts.append(property_desc)

            # Add domain context
            domain_classes = property_info.get('domainIncludes', [])
            if domain_classes:
                domain_str = ", ".join(domain_classes[:3])
                parts.append(f"Used with: {domain_str}")

            # Add range context
            range_classes = property_info.get('rangeIncludes', [])
            if range_classes:
                range_str = ", ".join(range_classes[:3])
                parts.append(f"Points to: {range_str}")

            # Combine all parts
            text = ". ".join(parts)

            embedding = self.embeddings.embed_text(
                text,
                input_type="search_document",
                rate_limit_delay=self.rate_limit_delay
            )
            self._property_embeddings_cache[property_name] = embedding

        return self._property_embeddings_cache[property_name]

    def _refine_with_constraints(self, result: Dict) -> Dict:
        """
        ENFORCE Schema.org domain/range constraints.

        The property's domain/range MUST be respected - this is not optional.
        If top candidate violates constraints, pick the first valid one.
        """
        matched_property = result['predicate']['matched_property']

        if not matched_property:
            return result

        property_info = self.ontology.get_property_info(matched_property)
        domain_classes = property_info.get('domainIncludes', [])
        range_classes = property_info.get('rangeIncludes', [])

        # ENFORCE subject based on domain constraints
        if domain_classes:
            subject_candidates = result['subject']['top_candidates']
            refined_subject = self._enforce_ontology_constraints(
                subject_candidates,
                domain_classes
            )

            if refined_subject:
                result['subject']['top_candidates'] = refined_subject
                result['subject']['matched_class'] = refined_subject[0][0]
                result['subject']['confidence'] = refined_subject[0][1]

        # ENFORCE object based on range constraints
        if range_classes:
            object_candidates = result['object']['top_candidates']
            refined_object = self._enforce_ontology_constraints(
                object_candidates,
                range_classes
            )

            if refined_object:
                result['object']['top_candidates'] = refined_object
                result['object']['matched_class'] = refined_object[0][0]
                result['object']['confidence'] = refined_object[0][1]

        return result

    def _enforce_ontology_constraints(
        self,
        candidates: List[Tuple[str, float]],
        allowed_classes: List[str]
    ) -> List[Tuple[str, float]]:
        """
        ENFORCE ontology constraints by filtering to only allowed classes.

        Returns only candidates that are in allowed_classes or their parent classes.
        """
        if not allowed_classes or not candidates:
            return candidates

        # Build set of allowed classes including parent classes
        allowed_set = set(allowed_classes)

        # Expand to include parent classes
        for class_name in allowed_classes:
            class_info = self.ontology.get_class_info(class_name)
            parent_classes = class_info.get('subClassOf', [])
            allowed_set.update(parent_classes)

        # Filter candidates to only valid ones
        valid_candidates = []
        for class_name, score in candidates:
            # Check if this class is allowed
            if class_name in allowed_set:
                valid_candidates.append((class_name, score))
                continue

            # Check if this class has a parent in allowed_set
            class_info = self.ontology.get_class_info(class_name)
            class_parents = class_info.get('subClassOf', [])
            if any(p in allowed_set for p in class_parents):
                valid_candidates.append((class_name, score))

        # If no valid candidates found, return original (edge case)
        if not valid_candidates:
            return candidates

        return valid_candidates

