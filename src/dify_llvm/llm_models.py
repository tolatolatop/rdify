from .openai_schemas import *
from typing import AsyncIterator, List, Union
from .apps.fake_llvm import fake_llm_stream

MODEL_REGISTRY = {
    "my-chat-model": {
        "id": "my-chat-model",
        "capabilities": {
            "chat": True,
            "completion": True,
            "stream": True,
        }
    },
    # 还可以加入更多模型
}


async def invoke_chat(model: str, messages: List[ChatMessage], stream: bool, **kwargs) -> AsyncIterator[ChatCompletionChoice]:
    return fake_llm_stream(messages)

async def invoke_completion(model: str, prompt: Union[str,List[str]], stream: bool, **kwargs) -> AsyncIterator[CompletionChoice]:
    return fake_llm_stream(prompt)
