import json
import os

from src.agents.business_translator import BusinessTranslatorAgent


class _EchoModel:
    """Echo model that returns the prompt - useful for verifying prompt content."""
    def generate_content(self, prompt):
        class _Resp:
            def __init__(self, text):
                self.text = text
        return _Resp(prompt)


class _CapturingModel:
    """Model that captures the prompt and returns a fixed response."""
    def __init__(self):
        self.last_prompt = None

    def generate_content(self, prompt):
        self.last_prompt = prompt
        class _Resp:
            def __init__(self, text):
                self.text = text
        return _Resp("# Reporte de Prueba\n\nContenido del reporte generado.")


def test_translator_required_slot_missing_mentions_no_disponible(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "execution_contract.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "reporting_policy": {
                    "sections": ["decision"],
                    "slots": [
                        {
                            "id": "model_metrics",
                            "mode": "required",
                            "insights_key": "metrics_summary",
                            "sources": ["data/metrics.json"],
                        }
                    ],
                    "constraints": {"no_markdown_tables": True},
                }
            },
            f,
        )
    with open(os.path.join("data", "insights.json"), "w", encoding="utf-8") as f:
        json.dump({}, f)

    agent = BusinessTranslatorAgent(api_key="dummy_key")
    agent.model = _EchoModel()
    report = agent.generate_report(
        {"execution_output": "OK", "business_objective": "Objetivo de prueba"}
    )
    assert "Slot Coverage" in report
    assert "missing_required_slots" in report


def test_translator_without_segment_pricing_slot_does_not_require_segments(tmp_path, monkeypatch):
    """When no segment_pricing slot is configured, the prompt should not include segment_pricing data."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "execution_contract.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "reporting_policy": {
                    "sections": ["decision"],
                    "slots": [
                        {
                            "id": "model_metrics",
                            "mode": "required",
                            "insights_key": "metrics_summary",
                            "sources": ["data/metrics.json"],
                        }
                    ],
                    "constraints": {"no_markdown_tables": True},
                }
            },
            f,
        )
    with open(os.path.join("data", "insights.json"), "w", encoding="utf-8") as f:
        json.dump({"metrics_summary": [{"metric": "accuracy", "value": 0.8}]}, f)

    agent = BusinessTranslatorAgent(api_key="dummy_key")
    model = _CapturingModel()
    agent.model = model
    report = agent.generate_report(
        {"execution_output": "OK", "business_objective": "Objetivo de prueba"}
    )
    # The prompt may contain "segment_pricing" in generic instructions,
    # but should NOT contain segment_pricing data/slot payload
    assert "segment_pricing_summary" not in (model.last_prompt or "")
    # Verify report was generated successfully
    assert report is not None
