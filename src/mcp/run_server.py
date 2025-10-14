"""Script per avviare il server MCP."""

import sys
from pathlib import Path

# Aggiungi il path del progetto al PYTHONPATH
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import ConfigManager
from src.mcp import MCPServer


def main():
    """Avvia il server MCP."""
    # Carica la configurazione
    config = ConfigManager()
    mcp_config = config.get_mcp_config()

    host = mcp_config.get('host', 'localhost')
    port = mcp_config.get('port', 8000)

    print(f"Starting MCP Server on {host}:{port}...")

    # Crea e avvia il server
    server = MCPServer(host=host, port=port)
    server.run(debug=True)


if __name__ == "__main__":
    main()
