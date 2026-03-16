from src.agents.results_advisor import ResultsAdvisorAgent


def test_generate_feature_engineering_advice_uses_contract_plan() -> None:
    advisor = ResultsAdvisorAgent(api_key="")
    advice = advisor.generate_feature_engineering_advice(
        {
            "baseline_metrics": {"model_performance": {"roc_auc": 0.7123}},
            "primary_metric_name": "roc_auc",
            "feature_engineering_plan": {
                "techniques": [{"technique": "interaction", "columns": ["x", "y"]}],
                "notes": "Mantener leakage guard",
            },
            "baseline_ml_script_snippet": "def train():\n  pass\n",
            "dataset_profile": {},
            "column_roles": {},
        }
    )
    assert "interaction" in advice.lower()
    assert "build_features" in advice
    assert "Mantener leakage guard".lower() in advice.lower()


def test_generate_feature_engineering_advice_fallback_is_universal_and_bounded() -> None:
    advisor = ResultsAdvisorAgent(api_key="")
    advice = advisor.generate_feature_engineering_advice(
        {
            "baseline_metrics": {"model_performance": {"roc_auc": 0.61}},
            "primary_metric_name": "roc_auc",
            "feature_engineering_plan": {"techniques": []},
            "dataset_profile": {
                "features_with_nulls": ["a"],
                "high_cardinality_columns": ["cat_col"],
            },
        }
    )
    lines = [line for line in advice.splitlines() if line.strip()]
    assert any("missing indicators" in line.lower() for line in lines)
    assert any("categorias raras" in line.lower() for line in lines)
    assert len(lines) <= 12


class _FakeGeminiResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeGeminiClient:
    def __init__(self, text: str = "", error: Exception | None = None) -> None:
        self._text = text
        self._error = error

    def generate_content(self, prompt: str):
        if self._error is not None:
            raise self._error
        return _FakeGeminiResponse(self._text)


def test_generate_feature_engineering_advice_hybrid_uses_llm_when_available(monkeypatch) -> None:
    monkeypatch.setenv("RESULTS_ADVISOR_FE_MODE", "hybrid")
    monkeypatch.delenv("MIMO_API_KEY", raising=False)

    def _fake_init(_api_key):
        return (
            "gemini",
            _FakeGeminiClient("- Edita build_features con tecnica A\n- Manten CV-safe fit"),
            "gemini-3-flash-preview",
            None,
        )

    monkeypatch.setattr("src.agents.results_advisor.init_reviewer_llm", _fake_init)

    advisor = ResultsAdvisorAgent()
    advice = advisor.generate_feature_engineering_advice(
        {
            "baseline_metrics": {"model_performance": {"roc_auc": 0.70}},
            "primary_metric_name": "roc_auc",
            "feature_engineering_plan": {"techniques": [{"technique": "interaction"}]},
            "baseline_ml_script_snippet": "def build_features(df):\n    return df\n",
            "dataset_profile": {},
            "column_roles": {},
        }
    )
    assert "tecnica A".lower() in advice.lower()
    assert advisor.last_fe_advice_meta.get("source") == "llm"
    assert advisor.last_fe_advice_meta.get("model") == "gemini-3-flash-preview"


def test_generate_feature_engineering_advice_hybrid_fallbacks_to_deterministic_on_llm_error(monkeypatch) -> None:
    monkeypatch.setenv("RESULTS_ADVISOR_FE_MODE", "hybrid")
    monkeypatch.delenv("MIMO_API_KEY", raising=False)

    def _fake_init(_api_key):
        return (
            "gemini",
            _FakeGeminiClient(error=RuntimeError("llm unavailable")),
            "gemini-3-flash-preview",
            None,
        )

    monkeypatch.setattr("src.agents.results_advisor.init_reviewer_llm", _fake_init)

    advisor = ResultsAdvisorAgent()
    advice = advisor.generate_feature_engineering_advice(
        {
            "baseline_metrics": {"model_performance": {"roc_auc": 0.70}},
            "primary_metric_name": "roc_auc",
            "feature_engineering_plan": {"techniques": [{"technique": "interaction"}]},
            "baseline_ml_script_snippet": "def train():\n    pass\n",
            "dataset_profile": {},
            "column_roles": {},
        }
    )
    assert "build_features" in advice
    assert advisor.last_fe_advice_meta.get("source") == "deterministic_fallback"


def test_results_advisor_prefers_own_client_for_llm_paths(monkeypatch) -> None:
    monkeypatch.setenv("RESULTS_ADVISOR_FE_MODE", "hybrid")
    monkeypatch.setenv("RESULTS_ADVISOR_CRITIQUE_MODE", "hybrid")

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def _unexpected_init(_api_key):
        raise AssertionError("init_reviewer_llm should not be used when ResultsAdvisor already has its own client")

    monkeypatch.setattr("src.agents.results_advisor.OpenAI", _FakeOpenAI)
    monkeypatch.setattr("src.agents.results_advisor.init_reviewer_llm", _unexpected_init)

    advisor = ResultsAdvisorAgent(api_key="test-key")

    assert advisor.fe_provider == "mimo"
    assert advisor.fe_model_name == "mimo-v2-flash"
    assert advisor.fe_client is advisor.client
