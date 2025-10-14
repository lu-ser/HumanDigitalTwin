"""
Agent che integra l'LLM con i tools MCP per interazioni autonome.
"""

from typing import Dict, Any, List, Optional
from src.llm.base_llm import BaseLLM
from src.mcp.mcp_tools import get_mcp_tools
from src.utils import get_logger
from src.prompts.prompt_manager import PromptManager


class MCPAgent:
    """
    Agent che permette all'LLM di interagire autonomamente con il server MCP.
    Utilizza function calling per permettere all'LLM di decidere quando
    recuperare dati dal server.
    """

    def __init__(
        self,
        llm: BaseLLM,
        mcp_base_url: str = "http://localhost:8000",
        system_prompt: Optional[str] = None,
        enable_logging: bool = True
    ):
        """
        Inizializza l'agent MCP.

        Args:
            llm: Istanza del servizio LLM
            mcp_base_url: URL base del server MCP
            system_prompt: Prompt di sistema personalizzato (opzionale)
            enable_logging: Abilita logging con Rich (default: True)
        """
        self.llm = llm
        self.mcp_base_url = mcp_base_url
        self.tools = get_mcp_tools(mcp_base_url)

        # System prompt di default
        self.system_prompt = system_prompt or self._get_default_system_prompt()

        # Storia della conversazione
        self.conversation_history: List[Dict[str, str]] = []

        # Logger
        self.enable_logging = enable_logging
        self.logger = get_logger() if enable_logging else None

    def _get_default_system_prompt(self) -> str:
        """Restituisce il system prompt di default per l'agent."""
        prompt_manager = PromptManager()
        return prompt_manager.get_system_prompt('mcp_agent')

    def reset_conversation(self) -> None:
        """Resetta la cronologia della conversazione."""
        self.conversation_history = []

    def chat(self, user_message: str, include_history: bool = True) -> Dict[str, Any]:
        """
        Invia un messaggio all'agent e ricevi una risposta.

        Args:
            user_message: Messaggio dell'utente
            include_history: Se True, include la cronologia della conversazione

        Returns:
            Dizionario con la risposta e metadati:
            {
                "response": str,
                "tools_used": List[str],
                "conversation_history": List[Dict]
            }
        """
        # Costruisci i messaggi
        messages = []

        # Aggiungi system prompt
        messages.append({
            "role": "system",
            "content": self.system_prompt
        })

        # Aggiungi cronologia se richiesto
        if include_history and self.conversation_history:
            messages.extend(self.conversation_history)

        # Aggiungi il messaggio corrente
        messages.append({
            "role": "user",
            "content": user_message
        })

        # Log messaggio utente
        if self.logger:
            self.logger.log_user_message(user_message)

        # Genera risposta usando function calling
        try:
            # Log chiamata LLM
            if self.logger:
                model_info = self.llm.get_model_info()

            response = self.llm.generate_with_tools(messages, self.tools)

            # Log risposta LLM
            if self.logger:
                self.logger.log_llm_call(messages, response, model_info)

            # Aggiorna la cronologia
            self.conversation_history.append({
                "role": "user",
                "content": user_message
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": response
            })

            # Log risposta agent
            if self.logger:
                self.logger.log_agent_response(response)

            return {
                "response": response,
                "tools_used": [],  # TODO: tracciare quali tools sono stati usati
                "conversation_history": self.conversation_history.copy()
            }

        except NotImplementedError as e:
            # Il provider LLM non supporta function calling
            if self.logger:
                self.logger.log_error(str(e), "Function calling not supported")

            return {
                "response": f"Errore: {str(e)}",
                "tools_used": [],
                "conversation_history": self.conversation_history.copy(),
                "error": str(e)
            }
        except Exception as e:
            if self.logger:
                self.logger.log_error(str(e), "Agent chat error")

            return {
                "response": f"Errore durante la generazione: {str(e)}",
                "tools_used": [],
                "conversation_history": self.conversation_history.copy(),
                "error": str(e)
            }

    def chat_stream(self, user_message: str) -> str:
        """
        Versione semplificata di chat che restituisce solo la risposta.

        Args:
            user_message: Messaggio dell'utente

        Returns:
            Risposta dell'agent
        """
        result = self.chat(user_message)
        return result.get("response", "Errore nella generazione della risposta")

    def get_available_tools(self) -> List[Dict[str, str]]:
        """
        Restituisce informazioni sui tools disponibili.

        Returns:
            Lista di dizionari con nome e descrizione dei tools
        """
        tools_info = []
        for tool in self.tools:
            tools_info.append({
                "name": tool.name,
                "description": tool.description
            })
        return tools_info

    def set_system_prompt(self, prompt: str) -> None:
        """
        Imposta un nuovo system prompt.

        Args:
            prompt: Nuovo system prompt
        """
        self.system_prompt = prompt
