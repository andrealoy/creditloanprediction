from types import SimpleNamespace

from main import MlflowPredictor, get_predictor, resolve_local_model_path, resolve_registered_model_uri


def test_resolve_local_model_path_uses_embedded_mlruns_artifacts():
    resolved = resolve_local_model_path("models:/m-61b9814452384517a217559a504a6193")

    assert resolved.endswith("mlruns/1/models/m-61b9814452384517a217559a504a6193/artifacts")


def test_resolve_registered_model_uri_prefers_latest_stage_version(monkeypatch):
    class FakeClient:
        def search_model_versions(self, query):
            assert query == "name = 'best_credit_loan_model'"
            return [
                SimpleNamespace(version="14", current_stage="Production", source="models:/m-older"),
                SimpleNamespace(version="16", current_stage="Production", source="models:/m-61b9814452384517a217559a504a6193"),
                SimpleNamespace(version="17", current_stage="Staging", source="models:/m-newer-staging"),
            ]

    monkeypatch.setattr("main.MlflowClient", lambda: FakeClient())

    resolved = resolve_registered_model_uri("best_credit_loan_model", stage="Production")

    assert resolved.endswith("mlruns/1/models/m-61b9814452384517a217559a504a6193/artifacts")


def test_get_predictor_defaults_to_mlflow(monkeypatch):
    captured = {}

    def fake_configure_mlflow():
        captured["configured"] = True

    def fake_resolve_registered_model_uri(model_name, stage=None, version=None):
        if model_name == "best_credit_loan_model":
            return "/tmp/mlflow-model"
        if model_name == "credit_loan_preprocessor":
            return "/tmp/mlflow-preprocessor"
        raise AssertionError(model_name)

    def fake_init(self, model_uri, preprocessor_uri=None):
        captured["model_uri"] = model_uri
        captured["preprocessor_uri"] = preprocessor_uri

    monkeypatch.delenv("MODEL_SOURCE", raising=False)
    monkeypatch.delenv("MLFLOW_USE_PREPROCESSOR", raising=False)
    monkeypatch.setattr("main.configure_mlflow", fake_configure_mlflow)
    monkeypatch.setattr("main.resolve_registered_model_uri", fake_resolve_registered_model_uri)
    monkeypatch.setattr("main.MlflowPredictor.__init__", fake_init)

    predictor = get_predictor()

    assert isinstance(predictor, MlflowPredictor)
    assert captured == {
        "configured": True,
        "model_uri": "/tmp/mlflow-model",
        "preprocessor_uri": None,
    }