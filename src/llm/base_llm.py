from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List


class BaseLLM(ABC):
    """Classe base astratta per tutti i servizi LLM."""

    def __init__(self, config: Dict[str, Any]):
        """
        Inizializza il servizio LLM.

        Args:
            config: Dizionario con la configurazione dell'LLM
        """
        self.config = config
        self.model = config.get('model')
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 2000)

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Genera una risposta dal prompt.

        Args:
            prompt: Il prompt da inviare all'LLM
            **kwargs: Parametri aggiuntivi specifici del provider

        Returns:
            La risposta generata dall'LLM
        """
        pass

    @abstractmethod
    def generate_with_history(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Genera una risposta considerando la cronologia dei messaggi.

        Args:
            messages: Lista di messaggi in formato [{"role": "user/assistant", "content": "..."}]
            **kwargs: Parametri aggiuntivi specifici del provider

        Returns:
            La risposta generata dall'LLM
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Verifica se il servizio LLM è disponibile.

        Returns:
            True se disponibile, False altrimenti
        """
        pass

    def generate_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Any],
        **kwargs
    ) -> str:
        """
        Genera una risposta usando function calling con tools esterni.

        Questo metodo è opzionale - non tutti i provider LLM supportano il function calling.
        L'implementazione di default solleva NotImplementedError.

        Args:
            messages: Lista di messaggi in formato [{"role": "user/assistant", "content": "..."}]
            tools: Lista di Langchain tools disponibili
            **kwargs: Parametri aggiuntivi

        Returns:
            La risposta generata dall'LLM dopo aver eventualmente chiamato i tools

        Raises:
            NotImplementedError: Se il provider non supporta function calling
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} non supporta function calling. "
            "Implementa questo metodo per abilitare l'uso di tools esterni."
        )

    def set_temperature(self, temperature: float) -> None:
        """Imposta la temperatura per la generazione."""
        self.temperature = temperature

    def set_max_tokens(self, max_tokens: int) -> None:
        """Imposta il numero massimo di token."""
        self.max_tokens = max_tokens

    def get_model_info(self) -> Dict[str, Any]:
        """
        Restituisce informazioni sul modello corrente.

        Returns:
            Dizionario con informazioni sul modello
        """
        return {
            'provider': self.__class__.__name__,
            'model': self.model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        }
