from .openai_schemas import *
from typing import AsyncIterator, List, Union
from .apps.fake_llvm import fake_llm_stream_chat, fake_llm_stream_completion

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
    "test-model": {
        "id": "test-model",
        "capabilities": {
            "chat": True,
            "completion": True,
            "stream": True,
        }
    }
}


async def invoke_chat(req: ChatCompletionRequest, **kwargs) -> AsyncIterator[ChatCompletionChoice]:
    return fake_llm_stream_chat(req)

async def invoke_completion(req: CompletionRequest, **kwargs) -> AsyncIterator[CompletionChoice]:
    return fake_llm_stream_completion(req)
