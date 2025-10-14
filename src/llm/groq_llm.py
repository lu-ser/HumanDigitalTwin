from typing import Dict, Any, List
from langchain_groq import ChatGroq
from .base_llm import BaseLLM
from ..utils import get_logger


class GroqLLM(BaseLLM):
    """Implementazione del servizio LLM usando Groq."""

    def __init__(self, config: Dict[str, Any], api_key: str):
        """
        Inizializza il servizio Groq.

        Args:
            config: Dizionario con la configurazione dell'LLM
            api_key: API key per Groq
        """
        super().__init__(config)
        self.api_key = api_key

        # Inizializza il client Groq tramite Langchain
        self.client = ChatGroq(
            groq_api_key=self.api_key,
            model_name=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Genera una risposta dal prompt usando Groq.

        Args:
            prompt: Il prompt da inviare all'LLM
            **kwargs: Parametri aggiuntivi (temperature, max_tokens, ecc.)

        Returns:
            La risposta generata dall'LLM
        """
        try:
            # Aggiorna parametri se forniti
            temperature = kwargs.get('temperature', self.temperature)
            max_tokens = kwargs.get('max_tokens', self.max_tokens)

            # Aggiorna il client con i nuovi parametri se necessario
            if temperature != self.temperature or max_tokens != self.max_tokens:
                self.client.temperature = temperature
                self.client.max_tokens = max_tokens

            # Invoca il modello
            response = self.client.invoke(prompt)

            return response.content

        except Exception as e:
            raise Exception(f"Errore nella generazione con Groq: {str(e)}")

    def generate_with_history(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Genera una risposta considerando la cronologia dei messaggi.

        Args:
            messages: Lista di messaggi in formato [{"role": "user/assistant/system", "content": "..."}]
            **kwargs: Parametri aggiuntivi

        Returns:
            La risposta generata dall'LLM
        """
        try:
            # Aggiorna parametri se forniti
            temperature = kwargs.get('temperature', self.temperature)
            max_tokens = kwargs.get('max_tokens', self.max_tokens)

            if temperature != self.temperature or max_tokens != self.max_tokens:
                self.client.temperature = temperature
                self.client.max_tokens = max_tokens

            # Converti i messaggi nel formato Langchain
            from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

            lc_messages = []
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')

                if role == 'system':
                    lc_messages.append(SystemMessage(content=content))
                elif role == 'assistant':
                    lc_messages.append(AIMessage(content=content))
                else:  # user
                    lc_messages.append(HumanMessage(content=content))

            # Invoca il modello
            response = self.client.invoke(lc_messages)

            return response.content

        except Exception as e:
            raise Exception(f"Errore nella generazione con cronologia (Groq): {str(e)}")

    def is_available(self) -> bool:
        """
        Verifica se il servizio Groq è disponibile.

        Returns:
            True se disponibile, False altrimenti
        """
        try:
            # Test semplice per verificare la disponibilità
            test_response = self.client.invoke("test")
            return True
        except Exception:
            return False

    def generate_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Any],
        **kwargs
    ) -> str:
        """
        Genera una risposta usando function calling con tools esterni.

        Groq supporta il binding di tools tramite Langchain.
        L'LLM può decidere autonomamente se e quando chiamare i tools.

        Args:
            messages: Lista di messaggi in formato [{"role": "user/assistant", "content": "..."}]
            tools: Lista di Langchain tools disponibili
            **kwargs: Parametri aggiuntivi

        Returns:
            La risposta generata dall'LLM dopo aver chiamato i tools se necessario
        """
        try:
            from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

            # Converti i messaggi nel formato Langchain
            lc_messages = []
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')

                if role == 'system':
                    lc_messages.append(SystemMessage(content=content))
                elif role == 'assistant':
                    lc_messages.append(AIMessage(content=content))
                else:  # user
                    lc_messages.append(HumanMessage(content=content))

            # Bind tools al modello
            llm_with_tools = self.client.bind_tools(tools)

            # Invoca il modello con i tools
            response = llm_with_tools.invoke(lc_messages)

            # Se ci sono tool calls, eseguili e richiama il modello
            if hasattr(response, 'tool_calls') and response.tool_calls:
                # Aggiungi la risposta del modello ai messaggi
                lc_messages.append(response)

                # Esegui ogni tool call
                from langchain_core.messages import ToolMessage

                for tool_call in response.tool_calls:
                    # Trova il tool corrispondente
                    tool_name = tool_call['name']
                    tool_args = tool_call['args']

                    # Cerca il tool nella lista
                    selected_tool = None
                    for tool in tools:
                        if tool.name == tool_name:
                            selected_tool = tool
                            break

                    if selected_tool:
                        # Esegui il tool
                        tool_result = selected_tool.invoke(tool_args)

                        # Log tool call
                        try:
                            logger = get_logger()
                            logger.log_tool_call(tool_name, tool_args, str(tool_result))
                        except:
                            pass  # Ignora errori di logging

                        # Aggiungi il risultato ai messaggi
                        lc_messages.append(
                            ToolMessage(
                                content=str(tool_result),
                                tool_call_id=tool_call['id']
                            )
                        )

                # Richiama il modello con i risultati dei tools
                final_response = llm_with_tools.invoke(lc_messages)
                return final_response.content
            else:
                # Nessun tool chiamato, restituisci la risposta diretta
                return response.content

        except Exception as e:
            raise Exception(f"Errore nella generazione con tools (Groq): {str(e)}")

    def get_model_info(self) -> Dict[str, Any]:
        """Restituisce informazioni sul modello Groq corrente."""
        info = super().get_model_info()
        info['provider'] = 'Groq'
        info['supports_tools'] = True
        return info
