import os
import logging

from dify_llvm.openai_schemas import *
from dify_llvm.models import ModelInterface, ModelInfo, ModelCapabilities, ModelRegistry
from pydify import ChatbotClient
from pydify.site import DifySite, DifyAppMode
from .schemas import DifySiteModel, DifyAppModel

logger = logging.getLogger("dify_llvm.apps.dify")

def get_config():
    return {
        "DIFY_SITE_URL": os.getenv("DIFY_SITE_URL"),
        "DIFY_BASE_URL": os.getenv("DIFY_BASE_URL"),
        "DIFY_APP_API_KEY": os.getenv("DIFY_APP_API_KEY"),
        "DIFY_EMAIL": os.getenv("DIFY_EMAIL"),
        "DIFY_PASSWORD": os.getenv("DIFY_PASSWORD"),
    }


DIFY_SITE_MODEL = DifySiteModel()

def get_site():
    config = get_config()
    site = DifySite(
        base_url=config["DIFY_SITE_URL"],
        email=config["DIFY_EMAIL"],
        password=config["DIFY_PASSWORD"],
    )
    return site


def get_client():
    site = get_site()
    client = ChatbotClient(site)
    return client


def parser_app_to_model_interface(app: DifyAppModel) -> ModelInterface:
    name = app.name
    model_info = ModelInfo(
        id=name,
        owned_by="dify",
        capabilities=ModelCapabilities(chat=True, completion=True, stream=True),
    )
    model_interface = ModelInterface(
        info=model_info,
        invoke_chat=invoke_chat,
        invoke_completion=invoke_completion,
    )
    return model_interface


def register_all_models(model_registry: ModelRegistry):
    logger.info("Registering all models")
    site = get_site()
    try:
        all_apps = site.fetch_all_apps()
        for app in all_apps:
            app_model = DifyAppModel(
                id=app['id'],
                name=app['name'],
                api_keys=[],
            )
            DIFY_SITE_MODEL.apps.append(app_model)

            logger.info(f"Registering model: {app_model.name}")
            model_interface = parser_app_to_model_interface(app_model)
            model_registry.register_model(app_model.name, model_interface)
    except Exception as e:
        logger.error(f"Error registering all models: {e}")


async def invoke_chat(req: ChatCompletionRequest):
    site = get_client()
    chatbot = site.get_chatbot(req.model)
    response = await chatbot.chat(req.messages, stream=req.stream, **req.model_dump())
    return response

async def invoke_completion(req: CompletionRequest):
    site = get_client()
    chatbot = site.get_chatbot(req.model)
    response = await chatbot.complete(req.prompt, stream=req.stream, **req.model_dump())
    return response
