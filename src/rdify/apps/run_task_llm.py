from .redirect_llm import redirect_llm_stream_chat
from ..openai_schemas import ChatCompletionRequest
from ..models import ModelInterface, ModelInfo, ModelCapabilities, ModelRegistry

async def run_task_llm_stream_chat(req: ChatCompletionRequest, **kwargs):
    async for chunk in redirect_llm_stream_chat(req, **kwargs):
        yield chunk

def register_run_task_llm(model_registry: ModelRegistry):
    model_registry.register_model("run-task-model", ModelInterface(
        info=ModelInfo(
            id="run-task-model",
            owned_by="self",
            capabilities=ModelCapabilities(chat=True, completion=True, stream=True),
        ),
        invoke_chat=run_task_llm_stream_chat,
        invoke_completion=None,
    ))
