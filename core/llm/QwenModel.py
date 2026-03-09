from typing import Any, ClassVar, Optional
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.base.llms.types import ChatMessage, ChatResponse
from llama_index.core.llms.callbacks import llm_completion_callback, llm_chat_callback
from openai import OpenAI


class QwenModel(CustomLLM):
    BASE_URL: ClassVar[str] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    api_key: str = "..."
    model_name: str = "qwen-plus"
    context_window: int = 120000
    max_tokens: int = 8000
    temperature: float = 0.7
    top_p: float = 0.8
    time_out: float = 300.0
    client: Any = None
    is_stream: bool = True
    input_token: int = 0
    total_token: int = 0

    def __init__(self,
                 api_key: str,
                 base_url: Optional[str] = None,
                 model_name: Optional[str] = None,
                 max_token: Optional[int] = None,
                 context_window: Optional[int] = None,
                 temperature: Optional[float] = None,
                 top_p: Optional[float] = None,
                 time_out: Optional[float] = None,
                 stream: Optional[bool] = None,
                 **kwargs):
        super().__init__(**kwargs)  # 调用父类构造函数
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
        self.is_stream = stream if stream is not None else self.is_stream

    def reinit_client(self):
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.BASE_URL,
        )

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
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
            model=self.model_name,  # 填写需要调用的模型编码
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stream=self.is_stream,
            timeout=self.time_out,
            # extra_body={"enable_thinking": True},
        )
        if not self.is_stream:
            completion_response = response.choices[0].message.content
            self.input_token += response.usage.prompt_tokens
            self.total_token += response.usage.total_tokens
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

    @llm_chat_callback()
    def chat(self, messages: list, **kwargs) -> ChatResponse:
        """
        真正的多角色 chat 调用，保留 system / user / assistant 角色。

        AutoLinkParser / AutoLinkOptimize 依赖此接口来正确传递 system prompt。
        若缺少此方法，llama_index 的 CustomLLM 基类会回退到 complete()，
        把所有角色拼成一条 user 消息，导致 system prompt 失效。

        其他所有 actor (DIN-SQL, CHESS, MACSQL …) 均使用 complete()，
        不受此方法影响。
        """
        api_messages = []
        for m in messages:
            # 兼容 llama_index ChatMessage 对象和普通 dict
            if hasattr(m, "role"):
                role    = m.role.value if hasattr(m.role, "value") else str(m.role)
                content = m.content or ""
            else:
                role    = m.get("role", "user")
                content = m.get("content", "")
            api_messages.append({"role": role, "content": content})

        if self.is_stream:
            full_text = ""
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=api_messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                stream=True,
                timeout=self.time_out,
            )
            for chunk in response:
                delta = chunk.choices[0].delta
                # 丢弃 reasoning_content，只收集正式回答内容
                if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                    pass
                else:
                    full_text += delta.content or ""
            text = full_text
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=api_messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                stream=False,
                timeout=self.time_out,
            )
            text = response.choices[0].message.content or ""

        return ChatResponse(
            message=ChatMessage(role="assistant", content=text)
        )
