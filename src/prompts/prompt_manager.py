import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class PromptManager:
    """Gestisce i prompt del sistema da file YAML."""

    def __init__(self, prompts_path: str = None):
        """
        Inizializza il PromptManager.

        Args:
            prompts_path: Percorso al file prompts.yaml. Se None, usa il path di default.
        """
        if prompts_path is None:
            # Percorso default: stesso directory di questo file
            prompts_path = Path(__file__).parent / "prompts.yaml"

        self.prompts_path = Path(prompts_path)
        self._prompts: Dict[str, Any] = {}

        # Carica i prompt
        self._load_prompts()

    def _load_prompts(self) -> None:
        """Carica i prompt dal file YAML."""
        if not self.prompts_path.exists():
            raise FileNotFoundError(f"File prompts non trovato: {self.prompts_path}")

        with open(self.prompts_path, 'r', encoding='utf-8') as f:
            self._prompts = yaml.safe_load(f)

    def get_prompt(self, prompt_name: str) -> Optional[Dict[str, str]]:
        """
        Recupera un prompt completo (system + user_template).

        Args:
            prompt_name: Nome del prompt

        Returns:
            Dizionario con 'system' e 'user_template' o None se non trovato
        """
        return self._prompts.get(prompt_name)

    def get_system_prompt(self, prompt_name: str) -> Optional[str]:
        """
        Recupera solo il system prompt.

        Args:
            prompt_name: Nome del prompt

        Returns:
            Il system prompt o None se non trovato
        """
        prompt = self._prompts.get(prompt_name)
        return prompt.get('system') if prompt else None

    def get_user_template(self, prompt_name: str) -> Optional[str]:
        """
        Recupera solo il template del prompt utente.

        Args:
            prompt_name: Nome del prompt

        Returns:
            Il template del prompt utente o None se non trovato
        """
        prompt = self._prompts.get(prompt_name)
        return prompt.get('user_template') if prompt else None

    def format_prompt(self, prompt_name: str, **kwargs) -> Optional[str]:
        """
        Formatta un prompt con i parametri forniti.

        Args:
            prompt_name: Nome del prompt
            **kwargs: Parametri da sostituire nel template

        Returns:
            Il prompt formattato o None se non trovato
        """
        template = self.get_user_template(prompt_name)
        if template:
            return template.format(**kwargs)
        return None

    def build_messages(self, prompt_name: str, **kwargs) -> list:
        """
        Costruisce una lista di messaggi per l'LLM.

        Args:
            prompt_name: Nome del prompt
            **kwargs: Parametri da sostituire nel template

        Returns:
            Lista di messaggi in formato [{"role": "system/user", "content": "..."}]
        """
        prompt = self.get_prompt(prompt_name)
        if not prompt:
            return []

        messages = []

        # Aggiungi system prompt se presente
        if 'system' in prompt and prompt['system']:
            messages.append({
                'role': 'system',
                'content': prompt['system'].strip()
            })

        # Aggiungi user prompt formattato
        if 'user_template' in prompt:
            formatted = prompt['user_template'].format(**kwargs)
            messages.append({
                'role': 'user',
                'content': formatted.strip()
            })

        return messages

    def list_prompts(self) -> list:
        """
        Restituisce la lista di tutti i prompt disponibili.

        Returns:
            Lista dei nomi dei prompt
        """
        return list(self._prompts.keys())

    def reload(self) -> None:
        """Ricarica i prompt dal file."""
        self._load_prompts()
