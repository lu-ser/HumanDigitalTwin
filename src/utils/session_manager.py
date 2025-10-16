"""
Session manager for saving and loading triplet extraction sessions.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any


class SessionManager:
    """Manages saving and loading of triplet extraction sessions."""

    def __init__(self, sessions_dir: str = "data/sessions"):
        """
        Initialize SessionManager.

        Args:
            sessions_dir: Directory for saving session files
        """
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def save_session(
        self,
        triplets: List[Dict[str, str]],
        metadata: Optional[Dict[str, Any]] = None,
        session_name: Optional[str] = None
    ) -> str:
        """
        Save a triplet extraction session to JSON file.

        Args:
            triplets: List of extracted triplets
            metadata: Optional metadata (input_text, chunk_size, etc.)
            session_name: Optional custom session name (default: timestamp)

        Returns:
            Path to saved session file
        """
        # Generate filename
        if session_name:
            # Sanitize session name
            safe_name = "".join(c for c in session_name if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_name = safe_name.replace(' ', '_')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{safe_name}.json"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"session_{timestamp}.json"

        filepath = self.sessions_dir / filename

        # Prepare session data
        session_data = {
            "timestamp": datetime.now().isoformat(),
            "triplets_count": len(triplets),
            "triplets": triplets,
            "metadata": metadata or {}
        }

        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)

        return str(filepath)

    def load_session(self, filepath: str) -> Dict[str, Any]:
        """
        Load a session from JSON file.

        Args:
            filepath: Path to session file

        Returns:
            Session data dictionary
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def list_sessions(self) -> List[Dict[str, str]]:
        """
        List all available sessions.

        Returns:
            List of session info dicts with 'filename', 'timestamp', 'triplets_count'
        """
        sessions = []

        for filepath in sorted(self.sessions_dir.glob("*.json"), reverse=True):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                sessions.append({
                    "filename": filepath.name,
                    "filepath": str(filepath),
                    "timestamp": data.get("timestamp", "Unknown"),
                    "triplets_count": data.get("triplets_count", 0),
                    "metadata": data.get("metadata", {})
                })
            except Exception as e:
                print(f"Error reading session {filepath}: {e}")
                continue

        return sessions

    def delete_session(self, filepath: str) -> bool:
        """
        Delete a session file.

        Args:
            filepath: Path to session file

        Returns:
            True if deleted, False otherwise
        """
        try:
            Path(filepath).unlink()
            return True
        except Exception as e:
            print(f"Error deleting session: {e}")
            return False
