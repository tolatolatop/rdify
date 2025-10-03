import os
import asyncio
import logging

from rdify.openai_schemas import *
from rdify.models import ModelInterface, ModelInfo, ModelCapabilities, ModelRegistry
from pydify import ChatbotClient
from pydify.site import DifySite, DifyAppMode
from .schemas import DifySiteModel, DifyAppModel
from .schemas import DifyEvent
from rdify.utils.thread_bridge import run_blocking_iter_in_thread

logger = logging.getLogger("rdify.apps.dify")

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


def get_or_create_new_api_key(model_name: str):
    logger.debug(f"Getting or creating new API key for model: {model_name}")
    app_model = DIFY_SITE_MODEL.get_app(model_name)
    if len(app_model.api_keys) > 0:
        return app_model.api_keys[0]
    site = get_site()
    app_api_keys = site.fetch_app_api_keys(app_model.id)
    if len(app_api_keys) == 0:
        logger.debug(f"Creating new API key for model: {model_name}")
        new_api_key = site.create_app_api_key(app_model.id)
        api_key = new_api_key['token']
    else:
        api_key = app_api_keys[0]['token']
    app_model.api_keys.append(api_key)
    return api_key

def get_client(model_name: str):
    config = get_config()
    api_key = get_or_create_new_api_key(model_name)
    client = ChatbotClient(
        api_key=api_key,
        base_url=config["DIFY_BASE_URL"],
    )
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


def fetch_all_apps():
    site = get_site()
    for app in site.fetch_all_apps():
        app_model = DifyAppModel(
            id=app['id'],
            name=app['name'],
            api_keys=[],
        )
        DIFY_SITE_MODEL.apps.append(app_model)
        logger.debug(f"Fetched app: {app_model.name}")
        yield app_model

def register_all_models(model_registry: ModelRegistry):
    logger.info("Registering all models")
    site = get_site()
    try:
        for app_model in fetch_all_apps():
            logger.info(f"Registering model: {app_model.name}")
            model_interface = parser_app_to_model_interface(app_model)
            model_registry.register_model(app_model.name, model_interface)
    except Exception as e:
        logger.error(f"Error registering all models: {e}")


async def invoke_chat(req: ChatCompletionRequest):
    client = get_client(req.model)
    last_message = req.messages[-1]
    content = last_message.content

    def _blocking_iter():
        # 这个生成器可能是阻塞的、同步的
        for chunk in client.send_message(
            query=content,
            user=req.user or "unknown",
            response_mode="streaming"
        ):
            yield DifyEvent.from_api_data(chunk)

    async for chunk in run_blocking_iter_in_thread(_blocking_iter):
        yield ChatCompletionChoice(
            index=0,
            message=ChatMessage(role="assistant", content=chunk.answer),
            finish_reason=None,
            delta=ChoiceDeltaContent(content=chunk.answer, role="assistant")
        )

async def invoke_completion(req: CompletionRequest):
    site = get_client(req.model)
    chatbot = site.get_chatbot(req.model)
    response = await chatbot.complete(req.prompt, stream=req.stream, **req.model_dump())
    return response
