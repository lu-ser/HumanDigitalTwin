from typing import Dict, Any
from .base_llm import BaseLLM
from .groq_llm import GroqLLM


class LLMFactory:
    """Factory per creare istanze di servizi LLM."""

    # Registry dei provider disponibili
    _providers = {
        'groq': GroqLLM,
        # Qui si possono aggiungere altri provider in futuro
        # 'openai': OpenAILLM,
        # 'anthropic': AnthropicLLM,
    }

    @classmethod
    def create(cls, provider: str, config: Dict[str, Any], api_key: str) -> BaseLLM:
        """
        Crea un'istanza del servizio LLM specificato.

        Args:
            provider: Nome del provider (es: 'groq', 'openai')
            config: Configurazione dell'LLM
            api_key: API key per il provider

        Returns:
            Istanza del servizio LLM

        Raises:
            ValueError: Se il provider non è supportato
        """
        provider_lower = provider.lower()

        if provider_lower not in cls._providers:
            available = ', '.join(cls._providers.keys())
            raise ValueError(
                f"Provider '{provider}' non supportato. "
                f"Provider disponibili: {available}"
            )

        llm_class = cls._providers[provider_lower]
        return llm_class(config, api_key)

    @classmethod
    def register_provider(cls, name: str, provider_class: type) -> None:
        """
        Registra un nuovo provider LLM.

        Args:
            name: Nome del provider
            provider_class: Classe che implementa BaseLLM
        """
        if not issubclass(provider_class, BaseLLM):
            raise TypeError(f"{provider_class} deve ereditare da BaseLLM")

        cls._providers[name.lower()] = provider_class

    @classmethod
    def get_available_providers(cls) -> list:
        """
        Restituisce la lista dei provider disponibili.

        Returns:
            Lista dei nomi dei provider disponibili
        """
        return list(cls._providers.keys())
