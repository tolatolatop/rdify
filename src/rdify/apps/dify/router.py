from fastapi import APIRouter
from fastapi.responses import JSONResponse
from .schemas import DifyOpenAICompatibleModel
from .extra_api import post_openai_compatible_models
from .extra_api import fetch_openai_compatible_models
from .core import get_site


dify_router = APIRouter(
    prefix="/dify",
    tags=["dify"],
)

@dify_router.post("/models/activate")
def activate_models():
    site = get_site()
    exist_models = fetch_openai_compatible_models(site)
    models = [
        DifyOpenAICompatibleModel(model="test-model"),
        DifyOpenAICompatibleModel(model="test-model-long-repeat"),
    ]
    for model in models:
        if model.model not in [m.model for m in exist_models]:
            post_openai_compatible_models(site, model.model_dump())
    return JSONResponse(content=[model.model_dump() for model in models])
