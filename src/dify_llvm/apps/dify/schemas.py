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
