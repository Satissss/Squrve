from typing import Any, ClassVar
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
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

    def __init__(self,
                 api_key: str,
                 model_name: str = None,
                 max_token: int = None,
                 context_window: int = None,
                 temperature: float = None,
                 top_p: float = None,
                 time_out: float = None,
                 stream: bool = None,
                 **kwargs):
        super().__init__(**kwargs)  # 调用父类构造函数
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.BASE_URL,
        )
        self.api_key = api_key
        self.model_name = self.model_name if not model_name else model_name
        self.temperature = self.temperature if not temperature else temperature
        self.top_p = self.top_p if not top_p else top_p
        self.max_tokens = self.max_tokens if not max_token else max_token
        self.context_window = self.context_window if not context_window else context_window
        self.time_out = self.time_out if not time_out else time_out
        self.is_stream = stream if stream else self.is_stream

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
            timeout=self.time_out
        )
        if not self.is_stream:
            completion_response = response.choices[0].message.content
            self.input_token += response.usage.prompt_tokens
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
    def stream_complete(
            self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        response = ""
        # 注意：这里使用了 dummy_response，但在类中没有定义
        # 你可能需要实现实际的流式响应逻辑
        dummy_response = ["这", "是", "一", "个", "测", "试", "响", "应"]
        for token in dummy_response:
            response += token
            yield CompletionResponse(text=response, delta=token)


if __name__ == "__main__":
    llm = QwenModel(
        api_key="sk-3f43effded4d49fc934829ee24e79f1d",
    )
    res = llm.complete("我的爸爸妈妈结婚为什么不叫我").text
    print(res)