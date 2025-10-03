import os

from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class DifyAppModel(BaseModel):
    id: str = Field(..., description="The ID of the app")
    name: str = Field(..., description="The name of the app")
    api_keys: List[str] = Field(..., description="The API keys of the app", default_factory=list)


class DifySiteModel(BaseModel):
    apps: List[DifyAppModel] = Field(..., description="The apps of the site", default_factory=list)

    def get_app(self, model_name: str) -> Optional[DifyAppModel]:
        return next((app for app in self.apps if app.name == model_name), None)

class ApiBaseModel(BaseModel):
    api_data: dict = Field(default_factory=dict, exclude=True)

    @classmethod
    def from_api_data(cls, data: dict):
        return cls(**data,  api_data=data)

class DifyEvent(ApiBaseModel):
    id: str = Field(description="事件ID")
    conversation_id: str = Field(description="会话ID")
    message_id: str = Field(description="消息ID")
    created_at: int = Field(description="创建时间")
    task_id: str = Field(description="任务ID")
    event: str = Field(description="事件")
    answer: str | None = Field(description="答案", default=None)
    thought: str | None = Field(description="思考", default=None)
    metadata: dict | None = Field(description="元数据", default=None)

class DifyResponse(ApiBaseModel):
    events: list[DifyEvent] = Field(description="数据")

    @classmethod
    def from_api_data(cls, data: list):
        events = [DifyEvent.from_api_data(item) for item in data]
        return cls(events=events,  api_data={"events": data})


class DifyLLMModel(ApiBaseModel):
    model: str = Field(description="模型名称")
    model_type: str = Field(description="模型类型")


def default_endpoint_url():
    return os.getenv("RDIFY_BASE_URL", "")

class DifyOpenAICompatibleModelCredentials(ApiBaseModel):
    mode: Literal["chat"] = Field(description="模式", default="chat")
    context_size: str = Field(description="上下文大小", default="40960")
    max_tokens_to_sample: str = Field(description="最大令牌数", default="40960")
    agent_though_support: str = Field(description="代理思考支持", default="supported")
    function_calling_type: str = Field(description="函数调用类型", default="function_call")
    stream_function_calling: str = Field(description="流函数调用", default="supported")
    vision_support: str = Field(description="视觉支持", default="no_support")
    structured_output_support: str = Field(description="结构化输出支持", default="not_supported")
    stream_mode_auth: str = Field(description="流模式认证", default="not_use")
    stream_mode_delimiter: str = Field(description="流模式分隔符", default="\n\n")
    voices: str = Field(description="声音", default="alloy")
    api_key: str = Field(description="API密钥", default="")
    endpoint_url: str = Field(description="端点URL", default=default_endpoint_url())


def default_load_balancing():
    return {
        "enabled": False,
        "configs": []
    }

class DifyOpenAICompatibleModel(ApiBaseModel):
    model: str = Field(description="模型名称")
    model_type: Literal["llm"] = Field(description="模型类型", default="llm")
    credentials: DifyOpenAICompatibleModelCredentials = Field(
        description="凭证",
        default_factory=DifyOpenAICompatibleModelCredentials
    )
    load_balancing: dict = Field(description="负载均衡", default_factory=default_load_balancing)
