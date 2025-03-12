import os
from openai import OpenAI

class OllamaClient(OpenAI):
    """Custom OpenAI client that routes requests to Ollama."""

    def __init__(self, model_name="mistral", **kwargs):
        # Configure to use Ollama's endpoint
        kwargs["base_url"] = "http://localhost:11434/v1"

        # Ollama doesn't require an API key but the client expects one
        kwargs["api_key"] = "ollama-placeholder-key"

        super().__init__(**kwargs)
        self.model_name = model_name
        
        # Check if the model exists
        print(f"Using Ollama model: {model_name}")

    def create_completion(self, *args, **kwargs):
        # Override model name if not explicitly provided
        if "model" not in kwargs:
            kwargs["model"] = self.model_name

        return super().create_completion(*args, **kwargs)

    def create_chat_completion(self, *args, **kwargs):
        # Override model name if not explicitly provided
        if "model" not in kwargs:
            kwargs["model"] = self.model_name

        return super().create_chat_completion(*args, **kwargs)
        
    # These methods are needed for compatibility with agents library
    def completion(self, prompt, **kwargs):
        if "model" not in kwargs:
            kwargs["model"] = self.model_name
        return self.completions.create(prompt=prompt, **kwargs)
        
    def chat_completion(self, messages, **kwargs):
        if "model" not in kwargs:
            kwargs["model"] = self.model_name
        return self.chat.completions.create(messages=messages, **kwargs)
