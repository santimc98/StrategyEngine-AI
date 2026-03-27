import json
import os

from src.agents.business_translator import BusinessTranslatorAgent


class _StructuredModel:
    def __init__(self, payload):
        self.payload = payload
        self.last_prompt = None

    def generate_content(self, prompt):
        self.last_prompt = prompt

        class _Resp:
            def __init__(self, text):
                self.text = text

        return _Resp(json.dumps(self.payload, ensure_ascii=False))


def _long_paragraph(seed: str) -> str:
    base = (
        f"{seed} Este bloque resume la situación ejecutiva, conecta la evidencia con el impacto de negocio "
        f"y evita afirmaciones vagas. La explicación prioriza la trazabilidad, la calidad de los datos, "
        f"los riesgos operativos y las decisiones que la dirección debe tomar a continuación."
    )
    return " ".join([base, base, base])


def _prepare_translator_workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "insights.json"), "w", encoding="utf-8") as f:
        json.dump({}, f)
    with open(os.path.join("data", "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"run_outcome": "NO_GO"}, f)


def test_translator_structured_layout_renders_optional_artifact_when_requested(tmp_path, monkeypatch):
    _prepare_translator_workspace(tmp_path, monkeypatch)

    payload = {
        "title": "Reporte Ejecutivo: Caso de Prueba",
        "blocks": [
            {"type": "heading", "level": 1, "text": "Reporte Ejecutivo: Caso de Prueba"},
            {"type": "heading", "level": 2, "text": "Decisión Ejecutiva"},
            {"type": "paragraph", "text": _long_paragraph("La decisión ejecutiva es NO_GO.")},
            {"type": "artifact", "artifact_key": "kpi_snapshot", "lead_in": "La síntesis KPI muestra la señal mínima para decidir.", "analysis": ["La tabla resume la señal ejecutiva clave y evita dispersar métricas irrelevantes."]},
            {"type": "heading", "level": 2, "text": "Riesgos y Limitaciones"},
            {"type": "bullet_list", "items": ["La calidad del activo es insuficiente para producción.", "La evidencia apunta a bloqueo operativo real."]},
            {"type": "heading", "level": 2, "text": "Siguientes Acciones"},
            {"type": "numbered_list", "items": ["Corregir el cuello de botella principal.", "Relanzar una validación controlada."]},
        ],
        "evidence": [{"claim": "La decisión se apoya en artefactos confirmados", "source": "data/run_summary.json -> run_outcome"}],
    }

    agent = BusinessTranslatorAgent(api_key="dummy_key")
    agent.model = _StructuredModel(payload)

    report = agent.generate_report(
        {"execution_output": "OK", "business_objective": "Evaluar si la solución está lista para pasar a producción"}
    )

    assert '<table class="exec-table kpi-snapshot">' in report
    assert isinstance(agent.last_report_blocks, list)
    assert any(block.get("type") == "artifact" and block.get("artifact_key") == "kpi_snapshot" for block in agent.last_report_blocks)
    assert any(block.get("type") == "markdown" and "## Evidencia usada" in str(block.get("content") or "") for block in agent.last_report_blocks)


def test_translator_structured_layout_does_not_force_kpi_table(tmp_path, monkeypatch):
    _prepare_translator_workspace(tmp_path, monkeypatch)

    payload = {
        "title": "Reporte Ejecutivo: Narrativa Pura",
        "blocks": [
            {"type": "heading", "level": 1, "text": "Reporte Ejecutivo: Narrativa Pura"},
            {"type": "heading", "level": 2, "text": "Decisión Ejecutiva"},
            {"type": "paragraph", "text": _long_paragraph("La decisión ejecutiva es NO_GO por falta de evidencia fiable.")},
            {"type": "heading", "level": 2, "text": "Hallazgos Clave"},
            {"type": "bullet_list", "items": ["El activo no es confiable.", "La ejecución no debe continuar sin reparar la causa raíz."]},
            {"type": "heading", "level": 2, "text": "Riesgos y Limitaciones"},
            {"type": "bullet_list", "items": ["Persisten riesgos de integridad.", "La señal sigue siendo insuficiente para negocio."]},
            {"type": "heading", "level": 2, "text": "Siguientes Acciones"},
            {"type": "numbered_list", "items": ["Corregir la fuente.", "Volver a validar."]},
        ],
        "evidence": [{"claim": "La conclusión proviene del resumen de la run", "source": "data/run_summary.json -> run_outcome"}],
    }

    agent = BusinessTranslatorAgent(api_key="dummy_key")
    agent.model = _StructuredModel(payload)

    report = agent.generate_report(
        {"execution_output": "OK", "business_objective": "Determinar si hay base para seguir invirtiendo en el pipeline"}
    )

    assert '<table class="exec-table kpi-snapshot">' not in report
    assert isinstance(agent.last_report_blocks, list)
    assert not any(block.get("type") == "artifact" and block.get("artifact_key") == "kpi_snapshot" for block in agent.last_report_blocks)
