import pickle
import logging
import os
from datetime import datetime
from functools import wraps
from rdify.config import logs_dir
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from rdify.openai_schemas import ChatCompletionRequest, ChatCompletionChoice, ChatMessage
from .redirect_llm import redirect_llm_stream_chat
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from ..openai_schemas import ChatCompletionRequest
from ..models import ModelInterface, ModelInfo, ModelCapabilities, ModelRegistry

logger = logging.getLogger('rdify')

class TaskIsFinishedResponse(BaseModel):
    is_finished: bool = Field(..., description="Whether the task is finished")
    message: str = Field(..., description="The message from the assistant")


def check_run_task_is_finished(task_log: str) -> TaskIsFinishedResponse:
    """
    使用ChatOpenAI检查日志
    """
    logger.debug(f"Checking if the task is finished: {len(task_log)}")
    llm = ChatOpenAI(model=os.getenv("MOONSHOT_MODEL"), temperature=0, base_url=os.getenv("MOONSHOT_URL"), api_key=os.getenv("MOONSHOT_API_KEY"))
    llm = llm.with_structured_output(TaskIsFinishedResponse)
    resp = llm.invoke(f"Check if the task is finished: <task_log>{task_log}</task_log>")
    logger.debug(f"Task {len(task_log)} is finished: {resp}")
    return resp


def map_message_to_string(message) -> str:
    output = ""
    if isinstance(message, ChatCompletionRequest):
        for message in message.messages:
            output += f"\n{message.role}: {message.content}"
    elif isinstance(message, ChatCompletionChoice):
        choice = message.choice[0]
        output += f"\n{choice.message.role}: {choice.message.content}"
    elif isinstance(message, ChatCompletionChunk):
        choice = message.choices[0]
        if choice.delta.role is not None:
            output += f"\n{choice.delta.role}: {choice.delta.content}"
        else:
            output += f"{choice.delta.content}"
    else:
        raise ValueError(f"Unsupported message type: {type(message)}")
    return output

def convert_conversation_to_task_log(conversation: list) -> str:
    """
    将会话转换为日志字符串
    """
    req = convert_conversation_to_chat_completion_request(conversation)
    output = "".join(map(map_message_to_string, [req]))
    return output

def check_conversation_is_finished(conversation: list) -> TaskIsFinishedResponse:
    """
    检查会话是否结束
    """
    task_log = convert_conversation_to_task_log(conversation)
    if task_log.strip().endswith("</tool_use>"):
        return TaskIsFinishedResponse(is_finished=True, message="Task is finished")
    return check_run_task_is_finished(task_log)


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


def convert_conversation_to_chat_completion_request(conversation: list) -> ChatCompletionRequest:
    """
    将会话转换为ChatCompletionRequest
    """
    messages = []
    if not isinstance(conversation[0], ChatCompletionRequest):
        raise ValueError("Conversation must start with a ChatCompletionRequest")
    buffer_message = None
    for message in conversation:
        if isinstance(message, ChatCompletionRequest):
            messages.extend(message.messages)
        elif isinstance(message, ChatCompletionChoice):
            messages.append(message.message)
        elif isinstance(message, ChatCompletionChunk):
            buffer_message = message_add_chunk(message, buffer_message)
            if buffer_message is not None and buffer_message not in messages:
                messages.append(buffer_message)
        else:
            raise ValueError(f"Unsupported message type: {type(message)}")
    resp = conversation[0].model_copy(update={"messages": messages}, deep=True)
    return resp

def message_add_chunk(chunk: ChatCompletionChunk, message: ChatMessage = None) -> ChatMessage:
    if message is None:
        if chunk.choices[0].delta.role is None:
            logger.warning("Role is required")
            return None
        message = ChatMessage(role=chunk.choices[0].delta.role, content=chunk.choices[0].delta.content)
    else:
        message.content += chunk.choices[0].delta.content or ""
    return message

def continue_stream(loop_count = 3):
    def decorator(func):
        @wraps(func)
        async def wrapper(req: ChatCompletionRequest, **kwargs):
            conversation = [req]
            task_is_finished = False
            for i in range(loop_count):
                logger.info(f"Continue stream loop {i}")
                async for chunk in func(req, **kwargs):
                    if isinstance(chunk, ChatCompletionChunk) and chunk.choices[0].finish_reason is not None:
                        resp = check_conversation_is_finished(conversation)
                        if resp.is_finished:
                            conversation.append(chunk)
                            task_is_finished = True
                            yield chunk
                            break
                        else:
                            req = convert_conversation_to_chat_completion_request(conversation)
                    else:
                        conversation.append(chunk)
                        yield chunk
                if task_is_finished:
                    break
        return wrapper
    return decorator

@dump_conversation
@continue_stream(loop_count=3)
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
