import importlib
import os
import sys

import pytest
from fastapi import HTTPException


def test_explain_returns_503_when_explainer_unavailable(monkeypatch):
    sys.path.insert(0, os.path.abspath("pylib"))
    sys.path.insert(0, os.path.abspath("."))
    import web_api

    web_api = importlib.reload(web_api)
    monkeypatch.setattr(web_api, "get_word_explanation", None)
    monkeypatch.setattr(web_api, "_word_explanation_import_error", "missing dependency")

    with pytest.raises(HTTPException) as exc:
        web_api.explain_word("hello")
    assert exc.value.status_code == 503
    assert "Word explanation service is unavailable." in str(exc.value.detail)
