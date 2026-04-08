import pytest

from main import get_predictor


@pytest.fixture(autouse=True)
def reset_app_state():
    get_predictor.cache_clear()
    yield
    get_predictor.cache_clear()