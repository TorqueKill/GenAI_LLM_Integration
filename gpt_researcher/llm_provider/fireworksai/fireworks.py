from langchain.llms import LLM
from langchain.llms.providers import Fireworks

class FireworksAIProvider(LLM):
    def __init__(self, api_key: str):
        self.fireworks = Fireworks(api_key=api_key)

    async def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7, **kwargs):
        """Generate text using Fireworks AI.

        Args:
            prompt (str): The input text prompt.
            max_tokens (int): The maximum number of tokens to generate.
            temperature (float): Sampling temperature.

        Returns:
            str: The generated text.
        """
        response = await self.fireworks.complete(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        return response.get("choices")[0].get("text", "")

    async def embed(self, texts: list[str]):
        """Generate embeddings for a list of texts.

        Args:
            texts (list[str]): The input texts to embed.

        Returns:
            list: The list of embeddings.
        """
        embeddings = await self.fireworks.embed(texts=texts)
        return embeddings
