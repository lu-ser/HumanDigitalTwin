"""
Predicate mappings for common natural language patterns.
Maps common verbs/phrases to Schema.org properties.
"""

from typing import Dict, List

# Common predicate mappings
PREDICATE_MAPPINGS: Dict[str, List[str]] = {
    # Identity/Occupation
    "is": ["jobTitle", "hasOccupation", "alumniOf", "memberOf", "additionalType"],
    "is a": ["jobTitle", "hasOccupation", "additionalType"],
    "is an": ["jobTitle", "hasOccupation", "additionalType"],
    "works as": ["jobTitle", "hasOccupation"],
    "works at": ["worksFor", "affiliation", "workLocation"],

    # Social relationships
    "knows": ["knows"],
    "friends with": ["knows"],
    "colleague of": ["colleague", "knows"],
    "spouse of": ["spouse"],
    "parent of": ["parent", "children"],
    "child of": ["children", "parent"],

    # Actions
    "drinks": ["agent"],  # For ConsumeAction
    "eats": ["agent"],
    "reads": ["agent"],
    "writes": ["author", "creator"],
    "creates": ["creator", "author"],
    "owns": ["owns"],
    "has": ["owns", "hasPart"],

    # Location
    "lives in": ["homeLocation", "address"],
    "located in": ["location", "contentLocation"],
    "born in": ["birthPlace"],
    "from": ["nationality", "birthPlace"],

    # Education
    "studied at": ["alumniOf", "affiliation"],
    "graduated from": ["alumniOf"],
    "attended": ["alumniOf", "attendee"],

    # Other
    "likes": ["interactionStatistic"],
    "interested in": ["knowsAbout"],
    "expert in": ["knowsAbout"],
}


def get_predicate_hints(predicate: str) -> List[str]:
    """
    Get Schema.org property hints for a natural language predicate.

    Args:
        predicate: Natural language predicate (e.g., "is", "works at")

    Returns:
        List of Schema.org property names to prioritize
    """
    predicate_lower = predicate.lower().strip()
    return PREDICATE_MAPPINGS.get(predicate_lower, [])
