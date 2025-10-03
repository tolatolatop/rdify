from dify_llvm.apps.dify.core import get_or_create_new_api_key, fetch_all_apps


def test_get_or_create_new_api_key():
    list(fetch_all_apps())
    api_key = get_or_create_new_api_key("测试应用")
    assert api_key is not None
    assert len(api_key) > 0
