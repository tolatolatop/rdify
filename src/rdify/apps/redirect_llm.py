import os
import json
import asyncio
import logging
from openai import OpenAI
from ..openai_schemas import ChatCompletionRequest, CompletionRequest, ChatCompletionChoice, CompletionChoice, ChoiceDeltaContent
from ..openai_schemas import ChatMessage
from ..models import ModelInterface, ModelInfo, ModelCapabilities, ModelRegistry


logger = logging.getLogger("rdify.apps.fake_llvm")

async def redirect_llm_stream(messages: list[ChatMessage]):
    client = OpenAI(api_key=os.getenv("MOONSHOT_API_KEY"), base_url=os.getenv("MOONSHOT_URL"))
    resp = client.chat.completions.create(
        model=os.getenv("MOONSHOT_MODEL"),
        messages=messages,
        stream=True
    )
    for chunk in resp:
        logger.debug(f"Redirecting chunk: {chunk}")
        yield chunk


async def redirect_llm_stream_chat(req: ChatCompletionRequest, **kwargs):
    logger.info(f"Redirecting chat: {req.messages}")
    async for chunk in redirect_llm_stream(req.messages):
        yield chunk


def register_redirect_llm(model_registry: ModelRegistry):
    logger.info("Registering redirect-model")
    model_registry.register_model("redirect-model", ModelInterface(
        info=ModelInfo(
            id="redirect-model",
            owned_by="self",
            capabilities=ModelCapabilities(chat=True, completion=True, stream=True),
        ),
        invoke_chat=redirect_llm_stream_chat,
        invoke_completion=None,
    ))
