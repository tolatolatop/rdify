from typing import AsyncIterator
import logging
import anyio

from .openai_schemas import *
from .models import ModelRegistry, ModelInterface
from .utils.cancel_scope import CancelScope

logger = logging.getLogger("rdify.llm_models")

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


def chat_event(req: ChatCompletionRequest, resp: ChatCompletionResponse, **kwargs):
    async def event_generator():
        # 你可以考虑先 yield 一个 “开头” 的 JSON（比如 id/model 信息），再逐 chunk 内容
        # 为简单起见，这里每个 chunk 直接 yield 一个 JSON 行
        scope_id = generate_id()
        with CancelScope(**kwargs) as cancel_scope:
            logger.debug(f"Chat event scope_id: {scope_id}")
            chunk_gen = await invoke_chat(req, cancel_scope=cancel_scope, **kwargs)
            async for chunk in chunk_gen:
                # chunk 是 ChatCompletionChoice 类型，其中 delta 不为 None
                resp.choices = [chunk]
                content = resp.model_dump_json()
                await cancel_scope.wait()
                yield "data: " + content + "\n\n"
                logger.debug(f"Chunk: {content}")
        msg = ChatCompletionChoice(
            index=0,
            message=ChatMessage(role="assistant", content=""),
            finish_reason="stop",
            delta=ChoiceDeltaContent(content="", role="assistant")
        )
        resp.choices = [msg]
        content = resp.model_dump_json()
        # 最后一个终止 chunk 可以带 finish_reason
        msg = ChatCompletionChoice(
            index=0,
            message=ChatMessage(role="assistant", content=""),
            finish_reason="stop",
            delta=ChoiceDeltaContent(content="", role="assistant")
        )
        resp.choices = [msg]
        content = resp.model_dump_json()
        yield "data: " + content + "\n\n"
        logger.debug(f"Finish chunk: {content}")
        yield "data: [DONE]\n\n"
    return event_generator


def completion_event(req: CompletionRequest, resp: CompletionResponse, **kwargs):
    async def event_generator():
        completion_gen = await invoke_completion(req, **kwargs)
        async for chunk in completion_gen:
            resp.choices = [chunk]
            content = resp.model_dump_json()
            yield "data: " + content + "\n\n"
            logger.debug(f"Chunk: {content}")
        msg = CompletionChoice(
            index=0,
            text="",
            finish_reason="stop"
        )
        resp.choices = [msg]
        content = resp.model_dump_json()
        yield "data: " + content + "\n\n"
        logger.debug(f"Finish chunk: {content}")
        yield "data: [DONE]\n\n"
    return event_generator

