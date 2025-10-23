import os
import pytest
from openai import OpenAI, OpenAIError

TEST_OPENAI_URL = os.getenv("MOONSHOT_URL", "https://api.moonshot.cn/v1")
TEST_API_KEY = os.getenv("MOONSHOT_API_KEY", "test_key")
TEST_MODEL = os.getenv("MOONSHOT_MODEL", "test-model")

@pytest.fixture
def client():
    # 创建一个 OpenAI 客户端指向你的服务
    # 如果你的服务用 API key 或认证，也可以在这里传入
    return OpenAI(api_key=TEST_API_KEY, base_url=TEST_OPENAI_URL)


def tool_infos() -> list[dict]:
    tool_info = {
        "type": "function",  # 必需：指定工具类型
        "function": {        # 必需：包含函数定义
            "name": "get_weather",
            "description": "Get the current weather conditions for a specified city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string", 
                        "description": "The name of the city to get weather information for"
                    }
                },
                "required": ["city"]
            }
        }
    }
    return [tool_info]

def test_get_toolcall(client: OpenAI):
    resp = client.chat.completions.create(
        model=TEST_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the weather in Beijing?"}
        ],
        tools=tool_infos(),
        temperature=0.5,
        stream=True,
        max_tokens=30000
    )

    cached_tool_calls = []
    for chunk in resp:
        chunk_dict = chunk.model_dump() if hasattr(chunk, "model_dump") else chunk
        assert "choices" in chunk_dict
        c = chunk_dict["choices"][0].get("delta")
        if c and "tool_calls" in c:
            cached_tool_calls.extend(c.get("tool_calls", None) or [])
    assert len(cached_tool_calls) > 0
    print(cached_tool_calls)
