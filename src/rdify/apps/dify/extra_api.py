import requests
from typing import List
from pydify.site import DifySite
from .schemas import DifyLLMModel

def post_openai_compatible_models(site: DifySite, model_config: dict):
    base_url = site.base_url
    access_token = site.access_token
    api_path = "console/api/workspaces/current/model-providers/langgenius/openai_api_compatible/openai_api_compatible/models"
    url = f"{base_url}/{api_path}"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    response = requests.post(url, headers=headers, json=model_config)
    if response.status_code != 200:
        response.raise_for_status()
    return response.json()


def fetch_llm_models(site: DifySite):
    base_url = site.base_url
    access_token = site.access_token
    api_path = "console/api/workspaces/current/models/model-types/llm"
    url = f"{base_url}/{api_path}"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        response.raise_for_status()
    return response.json()


def fetch_openai_compatible_models(site: DifySite) -> List[DifyLLMModel]:
    data = fetch_llm_models(site).get("data", [])
    for models in data:
        if models.get("provider") == "langgenius/openai_api_compatible/openai_api_compatible":
            return [DifyLLMModel.from_api_data(models) for models in models.get("models", [])]
    return []
