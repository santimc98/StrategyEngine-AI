"""Background worker that executes the LangGraph pipeline in a separate process.

Usage:
    python -m src.utils.background_worker <run_id>

The worker reads input from runs/<run_id>/worker_input.json,
executes the full graph pipeline, and writes status/log/final_state
files that Streamlit polls for UI updates.

This process is completely independent of Streamlit — it survives
session disconnects, browser closes, and tab backgrounding.
"""

import json
import os
import re
import sys
import time
import traceback

# Ensure project root is in path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.utils.paths import PROJECT_ROOT
os.chdir(PROJECT_ROOT)

from src.utils.run_status import (
    append_log,
    is_run_abort_requested,
    write_final_state,
    write_status,
)
from src.utils.sandbox_config import normalize_sandbox_config
from src.utils.sandbox_provider import (
    get_sandbox_class,
    get_sandbox_provider_spec,
)

# Progress weight per step (cumulative %)
_STEP_PROGRESS = {
    "steward": 12,
    "strategist": 22,
    "domain_expert": 34,
    "data_engineer": 48,
    "engineer": 60,
    "evaluate_results": 82,
    "translator": 94,
    "generate_pdf": 100,
}

_STAGE_NAMES = {
    "steward": "Auditando datos",
    "strategist": "Generando estrategias",
    "domain_expert": "Deliberacion experta",
    "data_engineer": "Limpieza de datos",
    "engineer": "Entrenando modelo ML",
    "evaluate_results": "Evaluando resultados",
    "translator": "Generando reporte",
    None: "Completado",
}


def _update_status(run_id, *, stage, progress, completed_steps,
                   iteration=0, max_iterations=6,
                   metric_name="", metric_value="",
                   status="running", error=None, started_at=None):
    write_status(
        run_id,
        status=status,
        stage=stage,
        stage_name=_STAGE_NAMES.get(stage, stage or "Procesando"),
        progress=progress,
        iteration=iteration,
        max_iterations=max_iterations,
        metric_name=metric_name,
        metric_value=metric_value,
        completed_steps=list(completed_steps),
        error=error,
        started_at=started_at,
    )


def main(run_id: str) -> None:
    input_path = os.path.join("runs", run_id, "worker_input.json")
    with open(input_path, "r", encoding="utf-8") as f:
        params = json.load(f)

    csv_path = params["csv_path"]
    business_objective = params["business_objective"]
    sandbox_config = normalize_sandbox_config(params.get("sandbox_config"))
    sandbox_provider = str(sandbox_config.get("provider") or "local").strip().lower() or "local"

    # Write initial status IMMEDIATELY — before any heavy imports.
    # This lets the Streamlit polling UI show progress right away.
    started_at = time.time()
    completed_steps: set = set()
    active_step = "steward"
    current_progress = 0
    ml_iteration = 0
    ml_max_iterations = 6
    best_metric_name = ""
    current_metric_value = ""

    _update_status(
        run_id, stage=active_step, progress=current_progress,
        completed_steps=completed_steps, started_at=started_at,
    )
    append_log(run_id, "Sistema", "Iniciando pipeline de analisis...", "info")
    append_log(run_id, "Sistema", "Cargando modulos (puede tardar unos minutos)...", "info")
    append_log(run_id, "Sistema", f"Sandbox seleccionado: {sandbox_provider}", "info")

    try:
        get_sandbox_class(sandbox_provider)
    except Exception as exc:
        spec = get_sandbox_provider_spec(sandbox_provider)
        append_log(run_id, "Sistema", f"Sandbox no disponible: {spec.label}", "error")
        _update_status(
            run_id,
            stage=active_step,
            progress=current_progress,
            completed_steps=completed_steps,
            iteration=ml_iteration,
            max_iterations=ml_max_iterations,
            metric_name=best_metric_name,
            metric_value=current_metric_value,
            status="error",
            error=str(exc),
            started_at=started_at,
        )
        return

    # Heavy imports — graph module is ~27K lines, takes 1-3 minutes to load
    overrides_path = os.path.join(PROJECT_ROOT, "data", "agent_model_overrides.json")
    if os.path.exists(overrides_path):
        try:
            with open(overrides_path, "r", encoding="utf-8") as f:
                overrides = json.load(f)
            if overrides:
                from src.graph.graph import set_runtime_agent_models
                set_runtime_agent_models(overrides)
                print(f"WORKER: Applied model overrides: {overrides}")
        except Exception as e:
            print(f"WORKER: Could not load model overrides: {e}")

    from src.graph.graph import app_graph

    append_log(run_id, "Sistema", "Modulos cargados. Ejecutando pipeline...", "ok")
    append_log(run_id, "Data Steward", "Analizando calidad e integridad de datos...", "info")

    initial_state = {
        "csv_path": csv_path,
        "business_objective": business_objective,
        "run_id": run_id,
        "sandbox_config": sandbox_config,
        "sandbox_provider": sandbox_provider,
    }

    final_state = initial_state.copy()

    try:
        for event in app_graph.stream(initial_state, config={"recursion_limit": 250}):
            if event is None:
                continue

            # Check file-based abort signal from Streamlit UI
            if is_run_abort_requested(run_id):
                from src.graph.graph import request_abort
                request_abort(f"abort_requested via UI for run {run_id}")
                append_log(run_id, "Sistema", "Ejecucion cancelada por el usuario.", "warn")
                write_final_state(run_id, final_state)
                _update_status(
                    run_id, stage=active_step, progress=current_progress,
                    completed_steps=completed_steps, iteration=ml_iteration,
                    max_iterations=ml_max_iterations, metric_name=best_metric_name,
                    metric_value=current_metric_value,
                    status="aborted", error="Cancelado por el usuario",
                    started_at=started_at,
                )
                print(f"WORKER: Run {run_id} aborted by user request")
                return

            for key, value in event.items():
                if value is not None:
                    final_state.update(value)

            if "steward" in event:
                completed_steps.add("steward")
                active_step = "strategist"
                current_progress = _STEP_PROGRESS["steward"]
                append_log(run_id, "Data Steward", "Auditoria de calidad completada.", "ok")
                append_log(run_id, "Strategist", "Generando 3 estrategias de alto impacto...", "info")

            elif "strategist" in event:
                completed_steps.add("strategist")
                active_step = "domain_expert"
                current_progress = _STEP_PROGRESS["strategist"]
                strategies = final_state.get("strategies", {})
                if isinstance(strategies, dict) and "strategies" in strategies:
                    strat_list = strategies["strategies"]
                    titles = [s.get("title", "?") for s in strat_list[:3]]
                    append_log(run_id, "Strategist", f"{len(strat_list)} estrategias generadas:", "ok")
                    for i, t in enumerate(titles, 1):
                        append_log(run_id, "Strategist", f"  {i}. {t}", "info")
                else:
                    append_log(run_id, "Strategist", "Estrategias generadas.", "ok")
                append_log(run_id, "Domain Expert", "Evaluando y puntuando cada estrategia...", "info")

            elif "domain_expert" in event:
                completed_steps.add("domain_expert")
                active_step = "data_engineer"
                current_progress = _STEP_PROGRESS["domain_expert"]
                selected = final_state.get("selected_strategy", {})
                reviews = final_state.get("domain_expert_reviews", [])
                if reviews:
                    for rev in reviews:
                        score = rev.get("score", "?")
                        title = rev.get("title", "?")
                        append_log(run_id, "Domain Expert", f"  {title} -- {score}/10", "info")
                sel_title = selected.get("title", "N/A") if isinstance(selected, dict) else "N/A"
                append_log(run_id, "Domain Expert", f"Estrategia ganadora: {sel_title}", "ok")
                append_log(run_id, "Data Engineer", "Ejecutando script de limpieza y estandarizacion...", "info")

            elif "data_engineer" in event:
                completed_steps.add("data_engineer")
                active_step = "engineer"
                current_progress = _STEP_PROGRESS["data_engineer"]
                append_log(run_id, "Data Engineer", "Dataset limpiado y estandarizado.", "ok")
                # Detect iteration policy from contract
                contract = final_state.get("execution_contract", {})
                if isinstance(contract, dict):
                    iter_policy = contract.get("iteration_policy", {})
                    if isinstance(iter_policy, dict):
                        limit = iter_policy.get("max_iterations", iter_policy.get("limit", ml_max_iterations))
                        if isinstance(limit, (int, float)) and limit >= 1:
                            ml_max_iterations = int(limit)
                    metric = contract.get("primary_metric", contract.get("metric", ""))
                    if metric:
                        best_metric_name = str(metric).upper()
                if not best_metric_name:
                    best_metric_name = "Metric"
                ml_iteration = 1
                append_log(run_id, "ML Engineer", f"Generando codigo -- Iteracion 1/{ml_max_iterations}...", "info")

            elif "engineer" in event:
                iteration = final_state.get("current_iteration", ml_iteration)
                if isinstance(iteration, int) and iteration > ml_iteration:
                    ml_iteration = iteration
                append_log(run_id, "ML Engineer", f"Codigo generado (Iteracion {ml_iteration}). Ejecutando...", "info")
                ml_range = _STEP_PROGRESS["evaluate_results"] - _STEP_PROGRESS["data_engineer"]
                iter_progress = min(ml_iteration / ml_max_iterations, 1.0)
                current_progress = _STEP_PROGRESS["data_engineer"] + int(ml_range * iter_progress * 0.7)

            elif "execute_code" in event:
                exec_output = str(final_state.get("execution_output", ""))
                for pattern in [
                    r"(?:RMSLE|rmsle)[:\s=]+([0-9]+\.?[0-9]*)",
                    r"(?:RMSE|rmse)[:\s=]+([0-9]+\.?[0-9]*)",
                    r"(?:MAE|mae)[:\s=]+([0-9]+\.?[0-9]*)",
                    r"(?:AUC|auc)[:\s=]+([0-9]+\.?[0-9]*)",
                    r"(?:F1|f1)[:\s=]+([0-9]+\.?[0-9]*)",
                    r"(?:Accuracy|accuracy)[:\s=]+([0-9]+\.?[0-9]*)",
                    r"(?:R2|r2|R-squared)[:\s=]+([0-9]+\.?[0-9]*)",
                ]:
                    match = re.search(pattern, exec_output, re.IGNORECASE)
                    if match:
                        current_metric_value = match.group(1)
                        if best_metric_name == "Metric":
                            name_match = re.search(
                                r"([A-Za-z0-9_-]+)[:\s=]+" + re.escape(current_metric_value),
                                exec_output,
                            )
                            if name_match:
                                best_metric_name = name_match.group(1).upper()
                        break
                metric_str = f" -- {best_metric_name}: {current_metric_value}" if current_metric_value else ""
                append_log(run_id, "ML Engineer", f"Ejecucion completada (Iteracion {ml_iteration}){metric_str}.", "ok")
                append_log(run_id, "Reviewer", "Evaluando resultados vs. objetivo de negocio...", "info")
                active_step = "evaluate_results"

            elif "evaluate_results" in event:
                verdict = final_state.get("review_verdict", "APPROVED")
                if verdict == "NEEDS_IMPROVEMENT":
                    feedback = final_state.get("execution_feedback", "")
                    if len(feedback) > 200:
                        feedback = feedback[:200] + "..."
                    append_log(run_id, "Reviewer", f"Requiere mejoras: {feedback}", "warn")
                    ml_iteration += 1
                    if ml_iteration <= ml_max_iterations:
                        append_log(run_id, "ML Engineer", f"Refinando modelo -- Iteracion {ml_iteration}/{ml_max_iterations}...", "info")
                    active_step = "engineer"
                    ml_range = _STEP_PROGRESS["evaluate_results"] - _STEP_PROGRESS["data_engineer"]
                    iter_progress = min(ml_iteration / ml_max_iterations, 1.0)
                    current_progress = _STEP_PROGRESS["data_engineer"] + int(ml_range * iter_progress)
                else:
                    completed_steps.add("engineer")
                    completed_steps.add("evaluate_results")
                    active_step = "translator"
                    current_progress = _STEP_PROGRESS["evaluate_results"]
                    metric_str = f" -- {best_metric_name}: {current_metric_value}" if current_metric_value else ""
                    append_log(run_id, "Reviewer", f"Resultados aprobados{metric_str}.", "ok")
                    append_log(run_id, "Translator", "Generando informe ejecutivo...", "info")

            elif "retry_handler" in event:
                pass

            elif "translator" in event:
                completed_steps.add("translator")
                active_step = None
                current_progress = _STEP_PROGRESS["translator"]
                append_log(run_id, "Translator", "Reporte ejecutivo generado.", "ok")

            elif "generate_pdf" in event:
                current_progress = _STEP_PROGRESS["generate_pdf"]
                append_log(run_id, "Sistema", "PDF final generado.", "ok")

            # Update status file after every event
            _update_status(
                run_id,
                stage=active_step,
                progress=current_progress,
                completed_steps=completed_steps,
                iteration=ml_iteration,
                max_iterations=ml_max_iterations,
                metric_name=best_metric_name,
                metric_value=current_metric_value,
                started_at=started_at,
            )

        # Pipeline completed successfully
        elapsed = time.time() - started_at
        m, s = divmod(int(elapsed), 60)
        h, m = divmod(m, 60)
        elapsed_str = f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
        append_log(run_id, "Sistema", f"Pipeline completado en {elapsed_str}.", "ok")

        write_final_state(run_id, final_state)
        _update_status(
            run_id,
            stage=None,
            progress=100,
            completed_steps=list({s[0] if isinstance(s, tuple) else s for s in completed_steps}),
            iteration=ml_iteration,
            max_iterations=ml_max_iterations,
            metric_name=best_metric_name,
            metric_value=current_metric_value,
            status="complete",
            started_at=started_at,
        )
        print(f"WORKER: Pipeline completed successfully for run {run_id}")

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        tb = traceback.format_exc()
        print(f"WORKER ERROR: {error_msg}\n{tb}")
        append_log(run_id, "Sistema", f"Error critico: {error_msg}", "warn")

        # Try to save whatever state we have
        try:
            write_final_state(run_id, final_state)
        except Exception:
            pass

        _update_status(
            run_id,
            stage=active_step,
            progress=current_progress,
            completed_steps=completed_steps,
            iteration=ml_iteration,
            max_iterations=ml_max_iterations,
            metric_name=best_metric_name,
            metric_value=current_metric_value,
            status="error",
            error=error_msg,
            started_at=started_at,
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.utils.background_worker <run_id>")
        sys.exit(1)
    main(sys.argv[1])
