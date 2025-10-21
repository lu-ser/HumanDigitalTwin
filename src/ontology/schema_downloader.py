"""
Utility to download Schema.org ontology JSON-LD file.
"""

import requests
from pathlib import Path
from typing import Optional


def download_schema_org(
    url: str = "https://schema.org/version/latest/schemaorg-current-https.jsonld",
    output_path: str = "data/ontology/schema.jsonld"
) -> bool:
    """
    Download Schema.org ontology JSON-LD file.
    
    Args:
        url: URL of the Schema.org JSON-LD file
        output_path: Path where to save the file
        
    Returns:
        True if download successful, False otherwise
    """
    output = Path(output_path)
    
    # Skip if already exists
    if output.exists():
        print(f"[Schema.org] File already exists at {output}")
        return True
    
    # Create parent directory
    output.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"[Schema.org] Downloading from {url}...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Save to file
        with open(output, "wb") as f:
            f.write(response.content)
        
        print(f"[Schema.org] Downloaded successfully to {output}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"[Schema.org] Download failed: {str(e)}")
        return False
    except Exception as e:
        print(f"[Schema.org] Unexpected error: {str(e)}")
        return False


def ensure_schema_org(config_manager) -> Optional[str]:
    """
    Ensure Schema.org ontology file is available.
    Downloads it if not present.

    Args:
        config_manager: ConfigManager instance

    Returns:
        Path to schema.jsonld file if available, None otherwise
    """
    ontology_config = config_manager.get_ontology_config()
    schema_url = ontology_config.get("schema_url")
    schema_path = ontology_config.get("schema_path")
    
    if not schema_url or not schema_path:
        print("[Schema.org] Missing configuration")
        return None
    
    # Download if needed
    if not Path(schema_path).exists():
        success = download_schema_org(schema_url, schema_path)
        if not success:
            return None
    
    return schema_path

