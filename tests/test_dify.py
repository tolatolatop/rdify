from rdify.apps.dify.core import get_or_create_new_api_key, fetch_all_apps, get_site
from rdify.apps.dify.extra_api import post_openai_compatible_models, fetch_llm_models
from rdify.apps.dify.extra_api import fetch_openai_compatible_models


def test_get_or_create_new_api_key():
    list(fetch_all_apps())
    api_key = get_or_create_new_api_key("测试应用")
    assert api_key is not None
    assert len(api_key) > 0


def test_post_openai_compatible_models():
    site = get_site()
    models = fetch_llm_models(site)
    assert 'data' in models

    models = fetch_openai_compatible_models(site)
    assert len(models) > 0
    model = models[0]
