"""
Sistema di logging con Rich per tracciare interazioni LLM e MCP.
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import json
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich import box


class AgentLogger:
    """Logger per tracciare interazioni agent, LLM e MCP con Rich e file."""

    def __init__(self, log_dir: str = "logs"):
        """
        Inizializza il logger.

        Args:
            log_dir: Directory per i file di log
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Crea un run ID unico
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"run_{self.run_id}.log"

        # Console Rich
        self.console = Console()

        # Contatori
        self.llm_calls = 0
        self.tool_calls = 0
        self.mcp_requests = 0

        # Scrivi header nel file
        self._write_to_file(f"=== Agent Run Started: {self.run_id} ===\n")
        self._write_to_file(f"Timestamp: {datetime.now().isoformat()}\n\n")

        # Mostra banner
        self._show_banner()

    def _show_banner(self):
        """Mostra banner iniziale."""
        banner = Text()
        banner.append("Human Digital Twin Agent\n", style="bold cyan")
        banner.append(f"Run ID: {self.run_id}\n", style="dim")
        banner.append(f"Log file: {self.log_file}", style="dim")

        self.console.print(Panel(banner, box=box.DOUBLE, border_style="cyan"))
        self.console.print()

    def _write_to_file(self, content: str):
        """Scrivi nel file di log."""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(content)

    def log_user_message(self, message: str):
        """
        Logga un messaggio utente.

        Args:
            message: Messaggio dell'utente
        """
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Console
        self.console.print(f"[bold blue][{timestamp}] User:[/bold blue]")
        self.console.print(Panel(message, border_style="blue", box=box.ROUNDED))
        self.console.print()

        # File
        self._write_to_file(f"\n[{timestamp}] USER MESSAGE:\n")
        self._write_to_file(f"{message}\n")
        self._write_to_file("-" * 80 + "\n")

    def log_llm_call(self, messages: list, response: str, model_info: Optional[Dict] = None):
        """
        Logga una chiamata all'LLM.

        Args:
            messages: Messaggi inviati all'LLM
            response: Risposta dell'LLM
            model_info: Info sul modello
        """
        self.llm_calls += 1
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Console
        self.console.print(f"[bold yellow][{timestamp}] LLM Call #{self.llm_calls}:[/bold yellow]")

        if model_info:
            info_text = f"Model: {model_info.get('model', 'N/A')} | Provider: {model_info.get('provider', 'N/A')}"
            self.console.print(f"[dim]{info_text}[/dim]")

        # Mostra messaggi
        self.console.print("[dim]Messages sent:[/dim]")
        for msg in messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            role_color = {"user": "blue", "assistant": "green", "system": "magenta"}.get(role, "white")
            self.console.print(f"  [{role_color}]{role}:[/{role_color}] {content[:100]}...")

        # Mostra risposta
        self.console.print("[dim]Response:[/dim]")
        self.console.print(Panel(response, border_style="yellow", box=box.ROUNDED))
        self.console.print()

        # File
        self._write_to_file(f"\n[{timestamp}] LLM CALL #{self.llm_calls}:\n")
        if model_info:
            self._write_to_file(f"Model: {json.dumps(model_info, indent=2)}\n")
        self._write_to_file("Messages:\n")
        self._write_to_file(json.dumps(messages, indent=2, ensure_ascii=False))
        self._write_to_file("\n\nResponse:\n")
        self._write_to_file(response)
        self._write_to_file("\n" + "-" * 80 + "\n")

    def log_tool_call(self, tool_name: str, tool_args: Dict[str, Any], tool_result: str):
        """
        Logga una chiamata a un tool.

        Args:
            tool_name: Nome del tool
            tool_args: Argomenti passati al tool
            tool_result: Risultato del tool
        """
        self.tool_calls += 1
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Console
        self.console.print(f"[bold magenta][{timestamp}] Tool Call #{self.tool_calls}:[/bold magenta]")

        # Crea tabella per gli argomenti
        table = Table(title=f"Tool: {tool_name}", box=box.SIMPLE)
        table.add_column("Argument", style="cyan")
        table.add_column("Value", style="white")

        for key, value in tool_args.items():
            table.add_row(key, str(value))

        self.console.print(table)

        # Mostra risultato
        self.console.print("[dim]Result:[/dim]")
        try:
            # Prova a fare pretty print del JSON
            result_json = json.loads(tool_result)
            syntax = Syntax(json.dumps(result_json, indent=2), "json", theme="monokai", line_numbers=False)
            self.console.print(syntax)
        except:
            # Se non è JSON, mostra come testo
            self.console.print(Panel(tool_result[:500], border_style="magenta", box=box.ROUNDED))

        self.console.print()

        # File
        self._write_to_file(f"\n[{timestamp}] TOOL CALL #{self.tool_calls}:\n")
        self._write_to_file(f"Tool: {tool_name}\n")
        self._write_to_file(f"Arguments: {json.dumps(tool_args, indent=2)}\n")
        self._write_to_file(f"Result:\n{tool_result}\n")
        self._write_to_file("-" * 80 + "\n")

    def log_mcp_request(self, method: str, endpoint: str, params: Optional[Dict] = None, response: Any = None):
        """
        Logga una richiesta al server MCP.

        Args:
            method: Metodo HTTP
            endpoint: Endpoint chiamato
            params: Parametri della richiesta
            response: Risposta del server
        """
        self.mcp_requests += 1
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Console
        self.console.print(f"[bold green][{timestamp}] MCP Request #{self.mcp_requests}:[/bold green]")
        self.console.print(f"  [cyan]{method}[/cyan] {endpoint}")

        if params:
            self.console.print(f"  [dim]Params: {params}[/dim]")

        if response:
            self.console.print("[dim]Response:[/dim]")
            try:
                syntax = Syntax(json.dumps(response, indent=2), "json", theme="monokai", line_numbers=False)
                self.console.print(syntax)
            except:
                self.console.print(f"  {str(response)[:200]}")

        self.console.print()

        # File
        self._write_to_file(f"\n[{timestamp}] MCP REQUEST #{self.mcp_requests}:\n")
        self._write_to_file(f"{method} {endpoint}\n")
        if params:
            self._write_to_file(f"Params: {json.dumps(params, indent=2)}\n")
        if response:
            self._write_to_file(f"Response: {json.dumps(response, indent=2)}\n")
        self._write_to_file("-" * 80 + "\n")

    def log_agent_response(self, response: str):
        """
        Logga la risposta finale dell'agent.

        Args:
            response: Risposta dell'agent
        """
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Console
        self.console.print(f"[bold green][{timestamp}] Agent Response:[/bold green]")
        self.console.print(Panel(response, border_style="green", box=box.ROUNDED))
        self.console.print()

        # File
        self._write_to_file(f"\n[{timestamp}] AGENT RESPONSE:\n")
        self._write_to_file(f"{response}\n")
        self._write_to_file("-" * 80 + "\n")

    def log_error(self, error: str, context: Optional[str] = None):
        """
        Logga un errore.

        Args:
            error: Messaggio di errore
            context: Contesto dell'errore
        """
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Console
        self.console.print(f"[bold red][{timestamp}] ERROR:[/bold red]")
        if context:
            self.console.print(f"[dim]Context: {context}[/dim]")
        self.console.print(Panel(error, border_style="red", box=box.HEAVY))
        self.console.print()

        # File
        self._write_to_file(f"\n[{timestamp}] ERROR:\n")
        if context:
            self._write_to_file(f"Context: {context}\n")
        self._write_to_file(f"{error}\n")
        self._write_to_file("-" * 80 + "\n")

    def log_summary(self):
        """Mostra un riassunto della sessione."""
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Crea tabella riassuntiva
        table = Table(title="Session Summary", box=box.DOUBLE_EDGE)
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="yellow", justify="right")

        table.add_row("LLM Calls", str(self.llm_calls))
        table.add_row("Tool Calls", str(self.tool_calls))
        table.add_row("MCP Requests", str(self.mcp_requests))

        self.console.print()
        self.console.print(table)
        self.console.print(f"\n[dim]Log file: {self.log_file}[/dim]\n")

        # File
        self._write_to_file(f"\n[{timestamp}] SESSION SUMMARY:\n")
        self._write_to_file(f"LLM Calls: {self.llm_calls}\n")
        self._write_to_file(f"Tool Calls: {self.tool_calls}\n")
        self._write_to_file(f"MCP Requests: {self.mcp_requests}\n")
        self._write_to_file("=" * 80 + "\n")


# Istanza globale del logger
_global_logger: Optional[AgentLogger] = None


def get_logger(log_dir: str = "logs") -> AgentLogger:
    """
    Ottiene l'istanza globale del logger (singleton).

    Args:
        log_dir: Directory per i log

    Returns:
        Istanza del logger
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = AgentLogger(log_dir)
    return _global_logger
