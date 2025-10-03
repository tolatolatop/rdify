import json
import asyncio
import logging
from ..openai_schemas import ChatCompletionRequest, CompletionRequest, ChatCompletionChoice, CompletionChoice, ChoiceDeltaContent
from ..openai_schemas import ChatMessage
from ..models import ModelInterface, ModelInfo, ModelCapabilities, ModelRegistry


logger = logging.getLogger("dify_llvm.apps.fake_llvm")

async def fake_llm_stream(prompt: str):
    """
    模拟 LLM 的逐步输出。
    实际情况你可以接本地模型、或者自己切片大文本。
    """
    text = f"<think> {prompt[::-1]} \n</think> {prompt[:]}"
    for i, ch in enumerate(text.split(" ")):
        yield ch
        await asyncio.sleep(0.05)  # 模拟生成延迟


async def fake_llm_stream_chat(req: ChatCompletionRequest):
    prompt = "".join([f"{message.role}: {message.content}\n" for message in req.messages])
    async for chunk in fake_llm_stream(prompt):
        yield ChatCompletionChoice(
            index=0,
            message=ChatMessage(role="assistant", content=chunk),
            finish_reason=None,
            delta=ChoiceDeltaContent(content=chunk, role="assistant")
        )


async def fake_llm_stream_completion(req: CompletionRequest):
    prompt = req.prompt if isinstance(req.prompt, str) else "\n".join(req.prompt)
    async for chunk in fake_llm_stream(prompt):
        yield CompletionChoice(
            index=0,
            text=chunk,
            finish_reason=None
        )


def register_fake_llvm(model_registry: ModelRegistry):
    logger.info("Registering test-model")
    model_registry.register_model("test-model", ModelInterface(
        info=ModelInfo(
            id="test-model",
            owned_by="self",
            capabilities=ModelCapabilities(chat=True, completion=True, stream=True),
        ),
        invoke_chat=fake_llm_stream_chat,
        invoke_completion=fake_llm_stream_completion,
    ))
