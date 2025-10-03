import asyncio
from typing import AsyncIterator, List, Union
import time
import uuid
import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from .openai_schemas import *
from .llm_models import MODEL_REGISTRY
from .llm_models import invoke_chat, invoke_completion

app = FastAPI()

# 假设你有一个内部适配器 / 接口，比如：
# async def invoke_chat(model: str, messages: List[ChatMessage], stream: bool, **kwargs) -> AsyncIterator[ChatCompletionChoice]
# async def invoke_completion(model: str, prompt: Union[str,List[str]], stream: bool, **kwargs) -> AsyncIterator[CompletionChoice]

# 模型列表 / 注册支持的模型（你可以动态加载）


@app.get("/v1/models", response_model=ListModelsResponse)
async def list_models():
    models = []
    for model_id, info in MODEL_REGISTRY.items():
        models.append(ModelInfo(id=model_id,
                                owned_by="self",
                                capabilities=info.get("capabilities", {})))
    return ListModelsResponse(data=models)

@app.get("/v1/models/{model_id}", response_model=GetModelResponse)
async def get_model(model_id: str):
    info = MODEL_REGISTRY.get(model_id)
    if not info:
        raise HTTPException(status_code=404, detail="Model not found")
    return GetModelResponse(id=model_id,
                            owned_by="self",
                            capabilities=info.get("capabilities", {}))

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    # 校验 model 是否支持 chat
    info = MODEL_REGISTRY.get(req.model)
    if not info or not info.get("capabilities", {}).get("chat", False):
        raise HTTPException(status_code=400, detail="Model not supported for chat")

    # 生成一个响应 id
    resp_id = str(uuid.uuid4())
    created = int(time.time())

    # 如果不是 stream 模式：一次性返回最终响应
    if not req.stream:
        # 收集所有 chunk
        chunks = []
        async for chunk in invoke_chat(
            req.model, req.messages, stream=False,
            temperature=req.temperature,
            top_p=req.top_p,
            max_tokens=req.max_tokens,
            stop=req.stop,
            **{}
        ):
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
        resp = ChatCompletionResponse(
            id=resp_id,
            created=created,
            model=req.model,
            choices=choices,
            usage=usage
        )
        return resp

    else:
        # stream=True 模式 — 返回 StreamingResponse，逐 chunk 推送
        async def event_generator():
            # 你可以考虑先 yield 一个 “开头” 的 JSON（比如 id/model 信息），再逐 chunk 内容
            # 为简单起见，这里每个 chunk 直接 yield 一个 JSON 行
            first = {
                "id": resp_id,
                "object": "chat.completion",
                "model": req.model,
                "created": created,
                # choices 会是一个数组，元素的 delta 部分逐步填充
                "choices": []
            }
            yield (json.dumps(first) + "\n").encode("utf-8")

            async for chunk in invoke_chat(req.model, req.messages, stream=True,
                                           temperature=req.temperature,
                                           top_p=req.top_p,
                                           max_tokens=req.max_tokens,
                                           stop=req.stop,
                                           **{}):
                # chunk 是 ChatCompletionChoice 类型，其中 delta 不为 None
                msg = {
                    "id": resp_id,
                    "object": "chat.completion",
                    "model": req.model,
                    "created": created,
                    "choices": [
                        {
                            "index": chunk.index,
                            "delta": {
                                "content": chunk.delta.content,
                                "role": chunk.delta.role,
                            },
                            "finish_reason": None
                        }
                    ]
                }
                yield (json.dumps(msg) + "\n").encode("utf-8")
            # 最后一个终止 chunk 可以带 finish_reason
            # 可以自定义发送一个标记结束
        return StreamingResponse(event_generator(), media_type="application/json")

@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    # 校验模型是否支持补全
    info = MODEL_REGISTRY.get(req.model)
    if not info or not info.get("capabilities", {}).get("completion", False):
        raise HTTPException(status_code=400, detail="Model not supported for completion")

    resp_id = str(uuid.uuid4())
    created = int(time.time())

    if not req.stream:
        chunks = []
        async for chunk in invoke_completion(req.model, req.prompt, stream=False,
                                              suffix=req.suffix,
                                              temperature=req.temperature,
                                              top_p=req.top_p,
                                              max_tokens=req.max_tokens,
                                              stop=req.stop,
                                              **{}):
            chunks.append(chunk)
        choices = chunks
        usage = Usage()
        resp = CompletionResponse(
            id=resp_id,
            created=created,
            model=req.model,
            choices=choices,
            usage=usage
        )
        return resp
    else:
        async def event_generator():
            # 同样，先可以发送初始 header
            first = {
                "id": resp_id,
                "object": "text_completion",
                "model": req.model,
                "created": created,
                "choices": []
            }
            yield (json.dumps(first) + "\n").encode("utf-8")

            async for chunk in invoke_completion(
                req.model, req.prompt, stream=True,
                suffix=req.suffix,
                temperature=req.temperature,
                top_p=req.top_p,
                max_tokens=req.max_tokens,
                stop=req.stop,
                **{}
            ):
                msg = {
                    "id": resp_id,
                    "object": "text_completion",
                    "model": req.model,
                    "created": created,
                    "choices": [
                        {
                            "index": chunk.index,
                            "text": chunk.text,
                            "finish_reason": None
                        }
                    ]
                }
                yield (json.dumps(msg) + "\n").encode("utf-8")
        return StreamingResponse(event_generator(), media_type="application/json")
