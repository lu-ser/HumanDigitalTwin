"""
Dataset export utilities.
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


def export_to_json(dataset: Dict[str, Any], output_dir: str = "output", filename: str = None) -> str:
    """
    Export dataset to JSON file.

    Args:
        dataset: Dataset dictionary
        output_dir: Output directory
        filename: Output filename (auto-generated if None)

    Returns:
        Path to saved file
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dataset_{timestamp}.json"

    filepath = output_path / filename

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    return str(filepath)


def export_to_csv(dataset: Dict[str, Any], output_dir: str = "output", filename: str = None) -> str:
    """
    Export dataset to CSV file (flattened).

    Args:
        dataset: Dataset dictionary
        output_dir: Output directory
        filename: Output filename (auto-generated if None)

    Returns:
        Path to saved file
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dataset_{timestamp}.csv"

    filepath = output_path / filename

    # Flatten dataset for CSV
    rows = []
    for scene in dataset.get("scenes", []):
        row = {
            "scene_id": scene.get("scene_id"),
            "day": scene.get("day"),
            "time": scene.get("time"),
            "description": scene.get("description"),
            # Device data (flatten)
            "device_data": json.dumps(scene.get("device_data", {})),
            # Calendar events count
            "calendar_events_count": len(scene.get("calendar_events", [])),
            # Messages count
            "messages_count": len(scene.get("messages", [])),
            # Deficiencies
            "physical_activity_status": scene.get("deficiencies", {}).get("physical_activity", {}).get("status"),
            "sleep_status": scene.get("deficiencies", {}).get("sleep", {}).get("status"),
            "nutrition_status": scene.get("deficiencies", {}).get("nutrition", {}).get("status"),
            "stress_status": scene.get("deficiencies", {}).get("stress", {}).get("status"),
            "hydration_status": scene.get("deficiencies", {}).get("hydration", {}).get("status"),
            "social_interaction_status": scene.get("deficiencies", {}).get("social_interaction", {}).get("status"),
        }
        rows.append(row)

    # Write CSV
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    return str(filepath)
