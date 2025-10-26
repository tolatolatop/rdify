import pickle
import logging
from datetime import datetime
from functools import wraps
from rdify.config import logs_dir
from .redirect_llm import redirect_llm_stream_chat
from ..openai_schemas import ChatCompletionRequest
from ..models import ModelInterface, ModelInfo, ModelCapabilities, ModelRegistry

logger = logging.getLogger(__name__)

def dump_conversation(func):
    @wraps(func)
    async def wrapper(req: ChatCompletionRequest, **kwargs):
        conversation_id = datetime.now().strftime("conversation_%Y%m%d%H%M%S.%f").replace(".", "_")
        conversation = [req]
        async for chunk in func(req, **kwargs):
            conversation.append(chunk)
            yield chunk
        try:
            with open(logs_dir / f"{conversation_id}.pkl", "wb") as f:
                pickle.dump(conversation, f)
        except Exception as e:
            logger.error(f"Error dumping conversation: {e}")
    return wrapper

@dump_conversation
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
