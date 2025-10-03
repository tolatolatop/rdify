import json
import asyncio
from fastapi.responses import StreamingResponse

async def fake_llm_stream(prompt: str):
    """
    模拟 LLM 的逐步输出。
    实际情况你可以接本地模型、或者自己切片大文本。
    """
    text = f"[深度思考后的回答] {prompt[::-1]}"
    for i, ch in enumerate(text):
        chunk = {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": {"content": ch if i > 0 else f"role: assistant\n{ch}"}
            }]
        }
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.05)  # 模拟生成延迟
    yield "data: [DONE]\n\n"
