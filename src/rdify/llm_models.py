from .openai_schemas import *
from typing import AsyncIterator
from .models import ModelRegistry, ModelInterface


MODEL_REGISTRY = ModelRegistry()

my_chat_model = ModelInterface(
    info=ModelInfo(
        id="my-chat-model",
        owned_by="self",
        capabilities=ModelCapabilities(
            chat=True,
            completion=True,
            stream=True,
        ),
    ),
    invoke_chat=None,
    invoke_completion=None,
)


MODEL_REGISTRY.register_model("my-chat-model", my_chat_model)

def register_model(model_id: str, model_info: ModelInterface):
    MODEL_REGISTRY.register_model(model_id, model_info)


async def invoke_chat(req: ChatCompletionRequest, **kwargs) -> AsyncIterator[ChatCompletionChoice]:
    return MODEL_REGISTRY.get_model_invoke_chat(req.model)(req, **kwargs)

async def invoke_completion(req: CompletionRequest, **kwargs) -> AsyncIterator[CompletionChoice]:
    return MODEL_REGISTRY.get_model_invoke_completion(req.model)(req, **kwargs)
