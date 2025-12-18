from typing import Any, ClassVar
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from openai import OpenAI


class GeminiModel(CustomLLM):
    BASE_URL: ClassVar[str] = "https://generativelanguage.googleapis.com/v1beta/openai/"

    api_key: str = "..."
    model_name: str = "gemini-1.5-pro"
    context_window: int = 200000
    max_tokens: int = 8000
    temperature: float = 0.7
    top_p: float = 0.8
    time_out: float = 300.0
    client: Any = None
    is_stream: bool = True
    input_token: int = 0

    def __init__(self,
                 api_key: str,
                 base_url: str | None,
                 model_name: str = None,
                 max_token: int = None,
                 context_window: int = None,
                 temperature: float = None,
                 top_p: float = None,
                 time_out: float = None,
                 stream: bool = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url if base_url else self.BASE_URL,
        )
        self.api_key = api_key
        self.model_name = self.model_name if not model_name else model_name
        self.temperature = self.temperature if not temperature else temperature
        self.top_p = self.top_p if not top_p else top_p
        self.max_tokens = self.max_tokens if not max_token else max_token
        self.context_window = self.context_window if not context_window else context_window
        self.time_out = self.time_out if not time_out else time_out
        self.is_stream = self.is_stream if stream is None else stream

    def reinit_client(self, base_url: str = None):
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url or self.BASE_URL,
        )

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_tokens,
            model_name=self.model_name,
        )

    def set_api_key(self, api_key: str):
        self.client.api_key = api_key

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stream=self.is_stream,
            timeout=self.time_out,
        )

        if not self.is_stream:
            completion_response = response.choices[0].message.content
            try:
                self.input_token += response.usage.prompt_tokens
            except Exception:
                pass
        else:
            completion_response = ""
            for chunk in response:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                    pass
                else:
                    completion_response += delta.content or ""

        return CompletionResponse(text=completion_response)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        accumulated_text = ""
        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stream=True,
            timeout=self.time_out,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            token = ""
            if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                token = ""
            else:
                token = delta.content or ""
            if token:
                accumulated_text += token
                yield CompletionResponse(text=accumulated_text, delta=token)

