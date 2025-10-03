from pydantic import BaseModel, Field
from typing import List, Optional

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
