import os

from colorama import Fore, Style
from langchain_fireworks.chat_models import ChatFireworks


class FireworksAIProvider:

    def __init__(
        self,
        model,
        temperature,
        max_tokens
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = self.get_api_key()
        self.llm = self.get_llm_model()

    def get_api_key(self):
        """
        Gets the Fireworks API key
        Returns:

        """
        try:
            api_key = os.environ["FIREWORKS_API_KEY"]
        except:
            raise Exception(
                "Fireworks API key not found. Please set the FIREWORKS_API_KEY environment variable.")
        return api_key

    def get_llm_model(self):
        # Initializing the chat model
        llm = ChatFireworks(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key
        )

        return llm

    def convert_messages(self, messages):
        """
        Converts messages based on their role into a format suitable for Fireworks.
        
        Args:
          messages: A list of dictionaries with keys 'role' and 'content'.
        
        Returns:
          A list of dictionaries formatted for Fireworks.
        """
        converted_messages = []
        for message in messages:
            if message["role"] == "system":
                converted_messages.append({"role": "system", "content": message["content"]})
            elif message["role"] == "user":
                converted_messages.append({"role": "user", "content": message["content"]})

        return converted_messages

    async def get_chat_response(self, messages, stream, websocket=None):
        if not stream:
            # Getting output from the model chain using ainvoke for asynchronous invoking
            converted_messages = self.convert_messages(messages)
            output = await self.llm.ainvoke(messages)


            return output.content

        else:
            return await self.stream_response(messages, websocket)

    async def stream_response(self, messages, websocket=None):
        paragraph = ""
        response = ""

        # Streaming the response using the chain astream method from langchain
        async for chunk in self.llm.astream(messages):
            content = chunk.content
            if content is not None:
                response += content
                paragraph += content
                if "\n" in paragraph:
                    if websocket is not None:
                        await websocket.send_json({"type": "report", "output": paragraph})
                    else:
                        print(f"{Fore.GREEN}{paragraph}{Style.RESET_ALL}")
                    paragraph = ""
                    
        return response

