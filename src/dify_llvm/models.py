from .openai_schemas import *
from typing import AsyncIterator, Callable, Dict
from dataclasses import dataclass


@dataclass
class ModelInterface:
    info: ModelInfo
    invoke_chat: Callable[[ChatCompletionRequest], AsyncIterator[ChatCompletionChoice]]
    invoke_completion: Callable[[CompletionRequest], AsyncIterator[CompletionChoice]]


@dataclass
class ModelRegistry:
    models: Dict[str, ModelInterface]

    def __init__(self):
        self.models = {}

    def register_model(self, model_id: str, model_info: ModelInterface):
        self.models[model_id] = model_info

    def get_model(self, model_id: str) -> ModelInterface:
        return self.models.get(model_id, None)

    def get_model_info(self, model_id: str) -> ModelInfo:
        if not self.models.get(model_id, None):
            return None
        return self.models[model_id].info
    
    def get_model_invoke_chat(self, model_id: str) -> Callable[[ChatCompletionRequest], AsyncIterator[ChatCompletionChoice]]:
        if not self.models.get(model_id, None):
            return None
        return self.models[model_id].invoke_chat
    
    def get_model_invoke_completion(self, model_id: str) -> Callable[[CompletionRequest], AsyncIterator[CompletionChoice]]:
        if not self.models.get(model_id, None):
            return None
        return self.models[model_id].invoke_completion

    def list_models(self) -> List[ModelInterface]:
        return [model for model in self.models.values()]
