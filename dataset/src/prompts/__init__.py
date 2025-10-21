"""Prompt management for dataset generation."""

import yaml
from pathlib import Path
from typing import Dict, Any, List


class PromptManager:
    """Manages prompts from YAML file for dataset generation."""

    def __init__(self, prompts_path: str = None):
        """
        Initialize PromptManager.

        Args:
            prompts_path: Path to prompts.yaml file
        """
        if prompts_path is None:
            prompts_path = Path(__file__).parent / "prompts.yaml"

        self.prompts_path = Path(prompts_path)
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> Dict[str, Any]:
        """Load prompts from YAML file."""
        with open(self.prompts_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def get_prompt(self, prompt_name: str) -> Dict[str, str]:
        """
        Get a specific prompt by name.

        Args:
            prompt_name: Name of the prompt

        Returns:
            Dictionary with 'system' and 'user_template' keys
        """
        if prompt_name not in self.prompts:
            raise ValueError(f"Prompt '{prompt_name}' not found in prompts.yaml")
        return self.prompts[prompt_name]

    def build_messages(
        self,
        prompt_name: str,
        **kwargs
    ) -> List[Dict[str, str]]:
        """
        Build message list for LLM from prompt template.

        Args:
            prompt_name: Name of the prompt
            **kwargs: Variables to substitute in user_template

        Returns:
            List of message dictionaries with 'role' and 'content'
        """
        prompt = self.get_prompt(prompt_name)

        messages = []

        # Add system message
        if 'system' in prompt:
            # Format system with kwargs if placeholders exist
            system_content = prompt['system'].format(**kwargs) if '{' in prompt['system'] else prompt['system']
            messages.append({
                'role': 'system',
                'content': system_content
            })

        # Add user message
        if 'user_template' in prompt:
            user_content = prompt['user_template'].format(**kwargs)
            messages.append({
                'role': 'user',
                'content': user_content
            })

        return messages
