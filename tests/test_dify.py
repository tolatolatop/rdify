import json
from rdify.apps.dify.core import get_or_create_new_api_key, fetch_all_apps, get_site
from rdify.apps.dify.extra_api import post_openai_compatible_models, fetch_llm_models
from rdify.apps.dify.extra_api import fetch_openai_compatible_models
from rdify.apps.dify.extra_api import delete_openai_compatible_models
from rdify.apps.dify.schemas import DifyOpenAICompatibleModel


def test_get_or_create_new_api_key():
    list(fetch_all_apps())
    api_key = get_or_create_new_api_key("测试应用")
    assert api_key is not None
    assert len(api_key) > 0


def test_openai_compatible_models():
    with open("tests/data/dify_trans.json", "r") as f:
        dify_trans = json.load(f)
    
    dify_model = DifyOpenAICompatibleModel.from_api_data(dify_trans)
    assert dify_model.model_dump() == dify_model.api_data

def test_post_openai_compatible_models():
    site = get_site()
    models = fetch_llm_models(site)
    assert 'data' in models

    models = fetch_openai_compatible_models(site)
    assert len(models) == 0

    dify_model = DifyOpenAICompatibleModel(model="test-model")
    post_openai_compatible_models(site, dify_model.model_dump())
    delete_openai_compatible_models(site, dify_model.model_dump())
