from .openai_schemas import *
from typing import AsyncIterator, List, Union
from .apps.fake_llvm import fake_llm_stream_chat, fake_llm_stream_completion

MODEL_REGISTRY = {
    "my-chat-model": ModelInfo(
        id="my-chat-model",
        owned_by="self",
        capabilities=ModelCapabilities(
            chat=True,
            completion=True,
            stream=True,
        ),
    ),
    "test-model": ModelInfo(
        id="test-model",
        owned_by="self",
        capabilities=ModelCapabilities(
            chat=True,
            completion=True,
            stream=True,
        ),
    ),
}

def register_model(model_id: str, model_info: ModelInfo):
    MODEL_REGISTRY[model_id] = model_info


async def invoke_chat(req: ChatCompletionRequest, **kwargs) -> AsyncIterator[ChatCompletionChoice]:
    return fake_llm_stream_chat(req)

async def invoke_completion(req: CompletionRequest, **kwargs) -> AsyncIterator[CompletionChoice]:
    return fake_llm_stream_completion(req)
