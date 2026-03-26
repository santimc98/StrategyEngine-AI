import pandas as pd

from src.agents import steward as steward_module
from src.agents.steward import StewardAgent


def test_steward_analyze_data_shape_initialized(tmp_path, monkeypatch):
    df = pd.DataFrame(
        {
            "Size": [10, 20, 30],
            "Debtors": [1, 2, 3],
            "Sector": ["A", "B", "C"],
            "1stYearAmount": [100.0, 200.0, 300.0],
            "CurrentPhase": ["Prospect", "Negotiation", "Contract"],
            "Probability": [0.1, 0.5, 0.9],
        }
    )
    csv_path = tmp_path / "sample.csv"
    df.to_csv(csv_path, index=False)

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    class DummyResponse:
        text = "DATA SUMMARY: ok"

    class DummyModel:
        model_name = "dummy"

        def generate_content(self, _prompt):
            return DummyResponse()

    monkeypatch.setattr(steward_module.genai, "GenerativeModel", lambda *args, **kwargs: DummyModel())
    monkeypatch.setattr(steward_module.genai, "configure", lambda **kwargs: None, raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    agent = StewardAgent(api_key=None)
    assert agent.provider == "gemini"
    result = agent.analyze_data(str(csv_path), business_objective="test")
    summary = result.get("summary", "")
    assert "shape not associated" not in summary


def test_steward_decide_semantics_pass1_uses_gemini_provider(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    responses = iter(
        [
            "not valid json",
            '{"primary_target":"target","split_candidates":["fold_id"],"id_candidates":["customer_id"],"evidence_requests":[]}',
        ]
    )

    class DummyResponse:
        def __init__(self, text):
            self.text = text

    class DummyModel:
        model_name = "dummy"

        def generate_content(self, _prompt):
            return DummyResponse(next(responses))

    monkeypatch.setattr(steward_module.genai, "GenerativeModel", lambda *args, **kwargs: DummyModel())
    monkeypatch.setattr(steward_module.genai, "configure", lambda **kwargs: None, raising=False)

    agent = StewardAgent(api_key=None)
    assert agent.provider == "gemini"

    result = agent.decide_semantics_pass1(
        {
            "business_objective": "Predict churn",
            "column_inventory_preview": {"head": ["customer_id", "target", "fold_id"]},
            "sample_rows": {"head": [{"customer_id": "1", "target": "yes", "fold_id": "train"}]},
        }
    )

    assert result["primary_target"] == "target"
    assert result["split_candidates"] == ["fold_id"]
    assert result["id_candidates"] == ["customer_id"]
