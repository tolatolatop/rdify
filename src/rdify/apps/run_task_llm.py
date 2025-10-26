import pickle
import logging
import os
from datetime import datetime
from functools import wraps
from rdify.config import logs_dir
from .redirect_llm import redirect_llm_stream_chat
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from ..openai_schemas import ChatCompletionRequest
from ..models import ModelInterface, ModelInfo, ModelCapabilities, ModelRegistry

logger = logging.getLogger(__name__)

class TaskIsFinishedResponse(BaseModel):
    is_finished: bool = Field(..., description="Whether the task is finished")
    message: str = Field(..., description="The message from the assistant")


def check_run_task_is_finished(task_log: str) -> TaskIsFinishedResponse:
    """
    使用ChatOpenAI检查日志
    """
    llm = ChatOpenAI(model=os.getenv("MOONSHOT_MODEL"), temperature=0, base_url=os.getenv("MOONSHOT_URL"), api_key=os.getenv("MOONSHOT_API_KEY"))
    llm = llm.with_structured_output(TaskIsFinishedResponse)
    resp = llm.invoke(f"Check if the task is finished: {task_log}")
    return resp


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
