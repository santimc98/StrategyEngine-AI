import json
import os

from src.agents.business_translator import (
    BusinessTranslatorAgent,
    _extract_first_json_object,
    _materialize_structured_report,
)


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


class _RawTextModel:
    def __init__(self, raw_text):
        self.raw_text = raw_text
        self.last_prompt = None

    def generate_content(self, prompt):
        self.last_prompt = prompt

        class _Resp:
            def __init__(self, text):
                self.text = text

        return _Resp(self.raw_text)


class _SequencedModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def generate_content(self, prompt):
        self.prompts.append(prompt)
        text = self.responses.pop(0) if self.responses else self.prompts[-1]

        class _Resp:
            def __init__(self, text):
                self.text = text

        return _Resp(text)


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


def test_extract_first_json_object_handles_arrays_before_trailing_text():
    text = (
        '{"title":"Reporte","blocks":['
        '{"type":"heading","level":1,"text":"Reporte"},'
        '{"type":"paragraph","text":"Resumen ejecutivo suficiente."}'
        "]}\n\n## Evidencia usada\n- data/run_summary.json"
    )

    payload = _extract_first_json_object(text)

    assert isinstance(payload, dict)
    assert payload["title"] == "Reporte"
    assert isinstance(payload["blocks"], list)
    assert len(payload["blocks"]) == 2


def test_materialize_structured_report_recovers_from_malformed_root_payload():
    malformed_payload = (
        '{"title":"Reporte Ejecutivo Recuperado","blocks":['
        '{"type":"heading","level":1,"text":"Reporte Ejecutivo Recuperado"},'
        '{"type":"heading","level":2,"text":"Decisión Ejecutiva"},'
        '{"type":"paragraph","text":"'
        + _long_paragraph("La decisión ejecutiva es NO_GO con limitaciones controladas.")
        + '"},'
        '{"type":"bullet_list","items":["La señal principal es insuficiente.","Persisten riesgos operativos."]}'
        ']},'
        '{"type":"artifact","artifact_key":"kpi_snapshot","lead_in":"La síntesis KPI condensa la decisión.","analysis":["La tabla materializa la señal ejecutiva relevante."]},'
        '{"type":"heading","level":2,"text":"Siguientes Acciones"},'
        '{"type":"numbered_list","items":["Corregir la causa raíz.","Relanzar la validación."]},'
        '{"claim":"La decisión se apoya en el resumen de la run","source":"data/run_summary.json -> run_outcome"}'
    )

    artifact_registry = {
        "kpi_snapshot": {
            "artifact_key": "kpi_snapshot",
            "artifact_type": "html_table",
            "title": "KPI Snapshot",
            "content_html": '<table class="exec-table kpi-snapshot"><tr><td>ok</td></tr></table>',
        }
    }

    report, blocks, payload, issues = _materialize_structured_report(
        content=malformed_payload,
        artifact_registry=artifact_registry,
        evidence_paths=["data/run_summary.json"],
        target_language_code="es",
    )

    assert issues == []
    assert isinstance(payload, dict)
    assert payload["title"] == "Reporte Ejecutivo Recuperado"
    assert isinstance(payload["blocks"], list)
    assert len(payload["blocks"]) >= 6
    assert "Reporte Ejecutivo Recuperado" in report
    assert '<table class="exec-table kpi-snapshot">' in report
    assert isinstance(blocks, list)
    assert any(block.get("type") == "artifact" and block.get("artifact_key") == "kpi_snapshot" for block in blocks)
    assert any(block.get("type") == "markdown" and "## Evidencia usada" in str(block.get("content") or "") for block in blocks)


def test_translator_structured_layout_repairs_missing_blocks_before_quality_pass(tmp_path, monkeypatch):
    _prepare_translator_workspace(tmp_path, monkeypatch)
    monkeypatch.setenv("TRANSLATOR_TWO_PASS_ENABLED", "0")

    repaired_payload = {
        "title": "Reporte Ejecutivo Reparado",
        "blocks": [
            {"type": "heading", "level": 1, "text": "Reporte Ejecutivo Reparado"},
            {"type": "heading", "level": 2, "text": "Decisión Ejecutiva"},
            {"type": "paragraph", "text": _long_paragraph("La decisión ejecutiva es NO_GO por evidencia insuficiente.")},
            {"type": "heading", "level": 2, "text": "Riesgos y Limitaciones"},
            {"type": "bullet_list", "items": ["Persisten riesgos de integridad.", "La calidad de evidencia sigue siendo insuficiente."]},
            {"type": "heading", "level": 2, "text": "Siguientes Acciones"},
            {"type": "numbered_list", "items": ["Corregir la causa raíz.", "Relanzar la validación controlada."]},
        ],
        "evidence": [{"claim": "La decisión se apoya en el resumen de la run", "source": "data/run_summary.json -> run_outcome"}],
    }

    agent = BusinessTranslatorAgent(api_key="dummy_key")
    agent.model = _SequencedModel(
        [
            json.dumps({"title": "Payload roto", "evidence": []}, ensure_ascii=False),
            json.dumps(repaired_payload, ensure_ascii=False),
        ]
    )

    report = agent.generate_report(
        {"execution_output": "OK", "business_objective": "Determinar si el pipeline está listo para producción"}
    )

    quality_check = json.loads((tmp_path / "data" / "translator_quality_check.json").read_text(encoding="utf-8"))

    assert "Reporte Ejecutivo Reparado" in report
    assert isinstance(agent.last_report_blocks, list)
    assert quality_check["repair_loop"]["total_attempts"] >= 2
    assert quality_check["repair_loop"]["history"][0]["score"] < 100
    assert quality_check["structured_layout"]["block_count"] > 0
    assert quality_check["quality_score"] >= quality_check["repair_loop"]["history"][0]["score"]
