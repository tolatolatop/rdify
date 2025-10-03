import os
import pytest
from openai import OpenAI, OpenAIError

TEST_OPENAI_URL = os.getenv("TEST_OPENAI_URL", "http://localhost:8000/v1")

@pytest.fixture
def client():
    # 创建一个 OpenAI 客户端指向你的服务
    # 如果你的服务用 API key 或认证，也可以在这里传入
    return OpenAI(api_key="test_key", base_url=TEST_OPENAI_URL)

def test_list_models(client):
    """
    测试 GET /v1/models 接口兼容性（新版 SDK 用法）
    """
    resp = client.models.list()
    # openai SDK 返回一个 Pydantic 模型对象，使用 model_dump 或 dict() 转成 dict
    resp_dict = resp.model_dump() if hasattr(resp, "model_dump") else resp
    assert "data" in resp_dict, f"No 'data' field in list-models response: {resp_dict}"
    data = resp_dict["data"]
    assert isinstance(data, list), f"'data' is not list: {data}"
    assert len(data) > 0, "Model list is empty"
    first = data[0]
    assert "id" in first, f"No id in first model entry: {first}"
    # OpenAI 官方 models 接口中 “object” 字段值通常为 “model”
    assert first.get("object") == "model", f"Expected object='model', got {first.get('object')}"

def test_get_model_by_id(client):
    """
    测试 GET /v1/models/{model_id} 接口兼容性
    """
    # 先 list 出一个模型 ID
    resp = client.models.list()
    resp_dict = resp.model_dump() if hasattr(resp, "model_dump") else resp
    data = resp_dict["data"]
    assert data, "No models returned by list => cannot test get by id"
    model_id = data[0]["id"]

    resp_single = client.models.retrieve(model_id)
    resp_single_dict = resp_single.model_dump() if hasattr(resp_single, "model_dump") else resp_single
    assert resp_single_dict.get("id") == model_id, f"Returned model id mismatch: {resp_single_dict}"
    assert resp_single_dict.get("object") == "model", f"Expected object='model', got {resp_single_dict.get('object')}"

def test_get_model_404_on_unknown(client):
    """
    测试请求不存在模型 ID 时是否返回 404 或符合 OpenAI 错误格式
    """
    bad_id = "nonexistent_model_abcdef"
    with pytest.raises(OpenAIError) as excinfo:
        _ = client.models.retrieve(bad_id)
    # 错误消息中应当包含 “404” 或 “not found”
    msg = str(excinfo.value).lower()
    assert "404" in msg or "not found" in msg, f"Expected 404 / not found, got: {msg}"

def test_chat_completion_basic(client):
    resp = client.chat.completions.create(
        model="test-model",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, who are you?"}
        ],
        temperature=0.5,
        stream=False,
        max_tokens=10
    )
    resp_dict = resp.model_dump() if hasattr(resp, "model_dump") else resp
    assert "choices" in resp_dict, f"No choices in chat completion response: {resp_dict}"
    choices = resp_dict["choices"]
    assert isinstance(choices, list) and len(choices) > 0
    msg = choices[0].get("message")
    assert msg is not None, f"No message in first choice: {choices[0]}"
    assert "role" in msg and "content" in msg, f"Message missing fields: {msg}"
    assert isinstance(msg["content"], str)

def test_chat_completion_streaming(client):
    gen = client.chat.completions.create(
        model="test-model",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, streaming test."}
        ],
        temperature=0.5,
        stream=True,
        max_tokens=10
    )
    saw_content = ""
    count = 0
    for chunk in gen:
        chunk_dict = chunk.model_dump() if hasattr(chunk, "model_dump") else chunk
        assert "choices" in chunk_dict, f"Chunk missing choices: {chunk_dict}"
        c = chunk_dict["choices"][0]
        if "delta" in c:
            delta = c["delta"]
            assert isinstance(delta, dict), f"Delta is not a dict: {delta}"
            if delta.get("content"):
                saw_content += delta["content"]
        else:
            # 有些实现可能把 full message 放在 streaming chunk
            msg = c.get("message")
            assert msg and "content" in msg
            saw_content += msg["content"]
        count += 1
        if count > 20:
            break
    assert count > 0, "No streaming chunks"
    assert len(saw_content) > 0, "Streaming produced no content"

def test_completions_basic(client):
    resp = client.completions.create(
        model="test-model",
        prompt="The capital of France is",
        temperature=0.0,
        max_tokens=5,
        stream=False
    )
    resp_dict = resp.model_dump() if hasattr(resp, "model_dump") else resp
    assert "choices" in resp_dict, f"No choices field: {resp_dict}"
    c0 = resp_dict["choices"][0]
    assert "text" in c0 and isinstance(c0["text"], str)

def test_completions_streaming(client):
    gen = client.completions.create(
        model="test-model",
        prompt="Once upon a time,",
        temperature=0.5,
        max_tokens=10,
        stream=True
    )
    saw_text = ""
    count = 0
    for chunk in gen:
        chunk_dict = chunk.model_dump() if hasattr(chunk, "model_dump") else chunk
        assert "choices" in chunk_dict
        c = chunk_dict["choices"][0]
        if "text" in c:
            saw_text += c["text"]
        count += 1
        if count > 20:
            break
    assert count > 0
    assert len(saw_text) > 0
