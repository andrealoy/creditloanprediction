import pytest

from main import app, get_predictor


@pytest.fixture(autouse=True)
def reset_app_state():
    app.dependency_overrides.clear()
    get_predictor.cache_clear()
    yield
    app.dependency_overrides.clear()
    get_predictor.cache_clear()