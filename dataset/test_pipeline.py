"""
Quick test script for the dataset generation pipeline.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Load env
load_dotenv()

from agents.dataset_generation_graph import DatasetGenerationGraph
from utils.exporter import export_to_json

def test_pipeline():
    """Test the complete pipeline."""

    # Check API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("[ERROR] GROQ_API_KEY not found in environment")
        return

    print("="*60)
    print("TESTING DATASET GENERATION PIPELINE")
    print("="*60)

    # Test profile
    profile = """
    Marco is a 32-year-old software engineer living in Milan, Italy.
    He works remotely from home most days, typically starting work at 9 AM
    and finishing around 6 PM. He enjoys running in the mornings 3 times a week
    and goes to the gym on weekends. Marco is generally health-conscious but
    sometimes skips breakfast when he has early meetings. He's sociable,
    often meeting friends for dinner or drinks on Friday evenings.
    He struggles with maintaining a consistent sleep schedule, especially
    when working on tight deadlines.
    """

    try:
        # Initialize graph
        print("\n[INIT] Initializing graph...")
        graph = DatasetGenerationGraph(
            llm_api_key=api_key,
            enable_logging=True
        )

        print("\n[SUCCESS] Graph initialized successfully!")
        print("[INFO] Graph diagram saved to assets/dataset_graph.png")

        # Generate dataset with 2 scenes for quick test
        print("\n[GENERATE] Generating dataset (2 scenes for quick test)...")
        result = graph.run(
            profile=profile,
            num_scenes=2
        )

        # Export
        print("\n[EXPORT] Exporting dataset...")
        output_dir = Path(__file__).parent / "output"
        filepath = export_to_json(result, output_dir=str(output_dir), filename="test_dataset.json")

        print(f"\n[SUCCESS] Dataset exported to: {filepath}")

        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Profile: {profile[:100]}...")
        print(f"Scenes generated: {len(result['scenes'])}")

        for scene in result['scenes']:
            print(f"\n  Scene {scene['scene_id']}: {scene['day']} {scene['time']}")
            print(f"    Description: {scene['description'][:80]}...")
            print(f"    Calendar events: {len(scene.get('calendar_events', []))}")
            print(f"    Messages: {len(scene.get('messages', []))}")

            # Deficiencies summary
            deficiencies = scene.get('deficiencies', {})
            issues = [k for k, v in deficiencies.items()
                     if k != 'scene_id' and isinstance(v, dict)
                     and v.get('status') in ['WARNING', 'INSUFFICIENT', 'HIGH']]
            if issues:
                print(f"    [WARNING] Health issues: {', '.join(issues)}")
            else:
                print(f"    [OK] No health issues detected")

        print("\n" + "="*60)
        print("[SUCCESS] TEST COMPLETED SUCCESSFULLY!")
        print("="*60)

    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()
