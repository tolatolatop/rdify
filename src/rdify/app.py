from contextlib import asynccontextmanager
import asyncio
from typing import AsyncIterator, List, Union
import time
import uuid
import json
import logging

from fastapi import FastAPI, HTTPException
from fastapi import Request
from fastapi.responses import StreamingResponse, JSONResponse
from .openai_schemas import *
from .llm_models import MODEL_REGISTRY
from .llm_models import invoke_chat, invoke_completion
from .config import config
from .apps.fake_llvm import register_fake_llvm
from .apps import dify, redirect_llm
from .llm_models import chat_event, completion_event

def register_all_models():
    logger.info("Registering all models")
    register_fake_llvm(MODEL_REGISTRY)
    dify.register_all_models(MODEL_REGISTRY)
    redirect_llm.register_redirect_llm(MODEL_REGISTRY)

@asynccontextmanager
async def lifespan(app: FastAPI):
    register_all_models()
    app.include_router(dify.router)
    yield


app = FastAPI(lifespan=lifespan)

logger = logging.getLogger("rdify")

# 假设你有一个内部适配器 / 接口，比如：
# async def invoke_chat(model: str, messages: List[ChatMessage], stream: bool, **kwargs) -> AsyncIterator[ChatCompletionChoice]
# async def invoke_completion(model: str, prompt: Union[str,List[str]], stream: bool, **kwargs) -> AsyncIterator[CompletionChoice]

# 模型列表 / 注册支持的模型（你可以动态加载）

@app.get("/v1/models/reload")
async def reload_models():
    logger.info("Reloading models")
    register_all_models()
    return JSONResponse(content={"message": "Models reloaded"})


@app.get("/v1/models", response_model=ListModelsResponse)
async def list_models():
    models = []
    for model in MODEL_REGISTRY.list_models():
        models.append(model.info)
    return ListModelsResponse(data=models)

@app.get("/v1/models/{model_id}", response_model=GetModelResponse)
async def get_model(model_id: str):
    info = MODEL_REGISTRY.get_model_info(model_id)
    if not info:
        raise HTTPException(status_code=404, detail="Model not found")
    return GetModelResponse(**info.model_dump())

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    logger.debug(f"body: {await request.body()}")
    logger.debug(f"ChatCompletionRequest: {req}")
    # 校验 model 是否支持 chat
    info = MODEL_REGISTRY.get_model_info(req.model)
    if not info or not info.capabilities.chat:
        raise HTTPException(status_code=400, detail="Model not supported for chat")

    resp = ChatCompletionResponse(
        model=req.model,
        choices=[]
    )

    context = {
        "request": request,
    }

    # 如果不是 stream 模式：一次性返回最终响应
    if not req.stream:
        # 收集所有 chunk
        chunks = []
        chunk_gen = await invoke_chat(req, context=context)
        async for chunk in chunk_gen:
            chunks.append(chunk)
        # 假设 chunks 最终只有一个完整 choice（你内部适配器可以决定如何组织）
        choices = []
        for idx, ch in enumerate(chunks):
            # ch 是 ChatCompletionChoice 实例（包含 message, finish_reason）
            choices.append(ch)
        usage = Usage(
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None
        )
        resp.choices = choices
        resp.usage = usage
        return resp

    else:
        # stream=True 模式 — 返回 StreamingResponse，逐 chunk 推送
        logger.debug(f"StreamingResponse: {req.model}")
        event_generator = chat_event(req, resp, context=context)
        return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/v1/completions")
async def completions(req: CompletionRequest, request: Request):
    logger.debug(f"CompletionRequest: {req}")
    # 校验模型是否支持补全
    info = MODEL_REGISTRY.get_model_info(req.model)
    if not info or not info.capabilities.completion:
        raise HTTPException(status_code=400, detail="Model not supported for completion")

    resp = CompletionResponse(
        model=req.model,
        choices=[]
    )

    context = {
        "request": request,
    }

    if not req.stream:
        chunks = []
        completion_gen = await invoke_completion(req, context=context)
        async for chunk in completion_gen:
            chunks.append(chunk)
        resp.choices = chunks
        resp.usage = Usage()
        return resp
    else:
        event_generator = completion_event(req, resp, context=context)
        return StreamingResponse(event_generator(), media_type="text/event-stream")
