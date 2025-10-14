import yaml
import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv


class ConfigManager:
    """Gestisce la configurazione del progetto da file YAML e variabili d'ambiente."""

    def __init__(self, config_path: str = None):
        """
        Inizializza il ConfigManager.

        Args:
            config_path: Percorso al file config.yaml. Se None, usa il path di default.
        """
        if config_path is None:
            # Percorso default: root del progetto
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config.yaml"

        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}

        # Carica le variabili d'ambiente
        load_dotenv()

        # Carica la configurazione
        self._load_config()

    def _load_config(self) -> None:
        """Carica la configurazione dal file YAML."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"File di configurazione non trovato: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Recupera un valore di configurazione usando la notazione dot.

        Args:
            key: Chiave in formato 'section.subsection.key'
            default: Valore di default se la chiave non esiste

        Returns:
            Il valore della configurazione o il default
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_env(self, key: str, default: str = None) -> str:
        """
        Recupera una variabile d'ambiente.

        Args:
            key: Nome della variabile d'ambiente
            default: Valore di default se non esiste

        Returns:
            Il valore della variabile d'ambiente o il default
        """
        return os.getenv(key, default)

    def get_llm_config(self) -> Dict[str, Any]:
        """Recupera la configurazione completa dell'LLM."""
        return self._config.get('llm', {})

    def get_mcp_config(self) -> Dict[str, Any]:
        """Recupera la configurazione del server MCP."""
        return self._config.get('mcp_server', {})

    def get_streamlit_config(self) -> Dict[str, Any]:
        """Recupera la configurazione di Streamlit."""
        return self._config.get('streamlit', {})

    def reload(self) -> None:
        """Ricarica la configurazione dal file."""
        self._load_config()
