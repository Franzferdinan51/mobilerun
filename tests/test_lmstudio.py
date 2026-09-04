import pytest

from mobilerun.agent.utils.llm_picker import load_llm


@pytest.mark.parametrize("provider", ["LMStudio", "lmstudio", "lm-studio", "lm studio"])
def test_lmstudio_uses_local_chat_and_never_inherits_openai_key(monkeypatch, provider):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    monkeypatch.delenv("LM_STUDIO_API_KEY", raising=False)
    monkeypatch.delenv("LM_STUDIO_BASE_URL", raising=False)
    llm = load_llm(provider, model="local-vision-model")
    assert llm.api_base == "http://localhost:1234/v1"
    assert llm.api_key == "lm-studio"
    assert llm.model == "local-vision-model"
    assert llm.metadata.is_chat_model
    assert not llm.metadata.is_function_calling_model
