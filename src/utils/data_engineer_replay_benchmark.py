import ast
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agents.data_engineer import DataEngineerAgent
from src.utils.contract_accessors import (
    get_required_outputs_by_owner,
    normalize_artifact_path,
)
from src.utils.data_engineer_preflight import data_engineer_preflight


def _load_json(path: Path, default: Any) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload
    except Exception:
        return default


def _first_existing(paths: List[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    text = text.strip("._-")
    return text or "default"


def _unique_ordered(values: List[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _collect_required_output_paths(contract: Dict[str, Any], de_view: Dict[str, Any]) -> List[str]:
    collected: List[str] = []
    for path in get_required_outputs_by_owner(contract, "data_engineer"):
        normalized = normalize_artifact_path(path)
        if normalized:
            collected.append(normalized)
    raw_outputs = de_view.get("required_outputs")
    if isinstance(raw_outputs, list):
        for item in raw_outputs:
            if isinstance(item, dict):
                normalized = normalize_artifact_path(
                    item.get("path") or item.get("output_path") or item.get("output")
                )
            else:
                normalized = normalize_artifact_path(item)
            if normalized:
                collected.append(normalized)
    output_path = normalize_artifact_path(de_view.get("output_path"))
    manifest_path = normalize_artifact_path(
        de_view.get("output_manifest_path") or de_view.get("manifest_path")
    )
    if output_path:
        collected.append(output_path)
    if manifest_path:
        collected.append(manifest_path)
    return _unique_ordered(collected)


def _extract_target_columns(contract: Dict[str, Any], strategy: Dict[str, Any]) -> List[str]:
    task_semantics = contract.get("task_semantics")
    if isinstance(task_semantics, dict):
        target_columns = task_semantics.get("target_columns")
        if isinstance(target_columns, list):
            return _unique_ordered([str(item) for item in target_columns])
        primary_target = str(task_semantics.get("primary_target") or "").strip()
        if primary_target:
            return [primary_target]
    top_level_targets = contract.get("target_columns")
    if isinstance(top_level_targets, list):
        return _unique_ordered([str(item) for item in top_level_targets])
    primary_target = str(contract.get("primary_target") or "").strip()
    if primary_target:
        return [primary_target]
    strategy_targets = strategy.get("target_columns")
    if isinstance(strategy_targets, list):
        return _unique_ordered([str(item) for item in strategy_targets])
    return []


def _build_column_sets_payload(case: "DataEngineerReplayCase") -> Dict[str, Any]:
    column_roles = case.execution_contract.get("column_roles")
    if not isinstance(column_roles, dict):
        column_roles = {}
    return {
        "required_columns": list(case.de_view.get("required_columns") or []),
        "model_features": list(case.model_features),
        "target_columns": list(case.target_columns),
        "optional_passthrough_columns": list(case.de_view.get("optional_passthrough_columns") or []),
        "column_roles": column_roles,
    }


def _read_csv_header(path: Path, sep: str, encoding: str) -> List[str]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding=encoding or "utf-8", newline="") as f:
            reader = csv.reader(f, delimiter=sep or ",")
            row = next(reader, [])
        return [str(item) for item in row]
    except Exception:
        return []


def _text_signals(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return " ".join(_text_signals(item) for item in value)
    if isinstance(value, dict):
        return " ".join(_text_signals(item) for item in value.values())
    return str(value).strip().lower()


def _candidate_enriched_paths(
    contract: Dict[str, Any],
    required_output_paths: List[str],
) -> List[str]:
    candidates: List[str] = []

    raw_outputs = contract.get("required_outputs")
    if isinstance(raw_outputs, list):
        for item in raw_outputs:
            if not isinstance(item, dict):
                continue
            path = normalize_artifact_path(item.get("path") or item.get("output_path") or item.get("output"))
            if not path:
                continue
            signals = " ".join(
                _text_signals(item.get(key))
                for key in ("intent", "description", "artifact", "name")
            )
            if any(
                token in signals
                for token in (
                    "enriched",
                    "enriquec",
                    "model_features",
                    "future_model",
                    "future_ml",
                    "handoff",
                )
            ):
                candidates.append(path)

    artifact_requirements = contract.get("artifact_requirements")
    if isinstance(artifact_requirements, dict):
        clean_dataset = artifact_requirements.get("clean_dataset")
        if isinstance(clean_dataset, dict):
            candidate = normalize_artifact_path(clean_dataset.get("output_path"))
            if candidate:
                candidates.append(candidate)

    fallback_tokens = (
        "leads_enriched_features.csv",
        "dataset_enriched.csv",
        "dataset_enriquecido.csv",
        "enriched",
        "enriquec",
    )
    for rel_path in required_output_paths:
        normalized = normalize_artifact_path(rel_path)
        if normalized and any(token in normalized.lower() for token in fallback_tokens):
            candidates.append(normalized)

    return _unique_ordered(candidates)


def _resolve_enriched_output_path(
    workspace: Path,
    contract: Dict[str, Any],
    required_output_paths: List[str],
) -> Optional[Path]:
    for rel_path in _candidate_enriched_paths(contract, required_output_paths):
        candidate = workspace / rel_path
        if candidate.exists():
            return candidate
    return None


def _load_benchmark_case_specs(repo_root: Path) -> Dict[str, Dict[str, Any]]:
    inventory_path = repo_root / "tests" / "data_engineer_benchmark_cases.json"
    payload = _load_json(inventory_path, {})
    cases = payload.get("cases") if isinstance(payload, dict) else None
    index: Dict[str, Dict[str, Any]] = {}
    if isinstance(cases, list):
        for item in cases:
            if not isinstance(item, dict):
                continue
            run_id = str(item.get("run_id") or "").strip()
            if run_id:
                index[run_id] = item
    return index


@dataclass
class DataEngineerReplayCase:
    run_id: str
    repo_root: Path
    run_dir: Path
    csv_path: Path
    business_objective: str
    csv_encoding: str
    csv_sep: str
    csv_decimal: str
    strategy: Dict[str, Any]
    execution_contract: Dict[str, Any]
    de_view: Dict[str, Any]
    data_audit: str
    required_output_paths: List[str]
    model_features: List[str]
    target_columns: List[str]
    prompt_path: Optional[Path]
    baseline_script_path: Optional[Path]
    baseline_error_path: Optional[Path]
    dataset_profile_path: Optional[Path]
    worker_input_path: Optional[Path]
    de_context_path: Optional[Path]
    quality_checks: List[Dict[str, Any]]

    def expected_enriched_columns(self) -> List[str]:
        return _unique_ordered(self.model_features + self.target_columns)


def load_data_engineer_replay_case(
    run_id: str,
    *,
    repo_root: Optional[str | Path] = None,
) -> DataEngineerReplayCase:
    repo = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
    run_dir = repo / "runs" / str(run_id or "").strip()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    worker_input_path = _first_existing([run_dir / "worker_input.json"])
    worker_input = _load_json(worker_input_path, {}) if worker_input_path else {}

    strategy_path = _first_existing(
        [
            run_dir / "work" / "data" / "strategy_spec.json",
            run_dir / "artifacts" / "data" / "strategy_spec.json",
        ]
    )
    strategy = _load_json(strategy_path, {}) if strategy_path else {}

    contract_path = _first_existing(
        [
            run_dir / "work" / "data" / "execution_contract_raw.json",
            run_dir / "artifacts" / "data" / "execution_contract_raw.json",
            run_dir / "agents" / "execution_planner" / "contract_raw.json",
        ]
    )
    execution_contract = _load_json(contract_path, {}) if contract_path else {}

    de_view_path = _first_existing(
        [
            run_dir / "work" / "data" / "contracts" / "views" / "de_view.json",
            run_dir / "artifacts" / "data" / "contracts" / "views" / "de_view.json",
        ]
    )
    de_view = _load_json(de_view_path, {}) if de_view_path else {}

    de_context_path = _first_existing(
        [
            run_dir / "work" / "artifacts" / "data_engineer_context.json",
            run_dir / "artifacts" / "data_engineer_context.json",
        ]
    )
    de_context = _load_json(de_context_path, {}) if de_context_path else {}
    data_audit = str(de_context.get("data_engineer_audit_override") or "").strip()

    prompt_path = _first_existing(
        [
            run_dir / "agents" / "data_engineer" / "iteration_1" / "prompt.txt",
            run_dir / "agents" / "data_engineer" / "iteration_2" / "prompt.txt",
        ]
    )
    baseline_script_path = _first_existing(
        [
            run_dir / "work" / "artifacts" / "data_engineer_last.py",
            run_dir / "artifacts" / "data_engineer_last.py",
        ]
    )
    baseline_error_path = _first_existing(
        [
            run_dir / "work" / "artifacts" / "data_engineer_sandbox_last_error.json",
            run_dir / "artifacts" / "data_engineer_sandbox_last_error.json",
        ]
    )
    dataset_profile_path = _first_existing(
        [
            run_dir / "work" / "data" / "dataset_profile.json",
            run_dir / "artifacts" / "data" / "dataset_profile.json",
        ]
    )

    csv_path_raw = str(worker_input.get("csv_path") or "").strip()
    csv_path = Path(csv_path_raw)
    if not csv_path.is_absolute():
        csv_path = (repo / csv_path).resolve()

    model_features = execution_contract.get("model_features")
    if not isinstance(model_features, list):
        model_features = de_view.get("model_features")
    if not isinstance(model_features, list):
        model_features = []
    target_columns = _extract_target_columns(execution_contract, strategy)
    case_specs = _load_benchmark_case_specs(repo)
    case_spec = case_specs.get(str(run_id).strip(), {})

    return DataEngineerReplayCase(
        run_id=str(run_id),
        repo_root=repo,
        run_dir=run_dir,
        csv_path=csv_path,
        business_objective=str(worker_input.get("business_objective") or ""),
        csv_encoding=str(worker_input.get("csv_encoding") or de_context.get("csv_encoding") or "utf-8"),
        csv_sep=str(worker_input.get("csv_sep") or de_context.get("csv_sep") or ","),
        csv_decimal=str(worker_input.get("csv_decimal") or de_context.get("csv_decimal") or "."),
        strategy=strategy if isinstance(strategy, dict) else {},
        execution_contract=execution_contract if isinstance(execution_contract, dict) else {},
        de_view=de_view if isinstance(de_view, dict) else {},
        data_audit=data_audit,
        required_output_paths=_collect_required_output_paths(
            execution_contract if isinstance(execution_contract, dict) else {},
            de_view if isinstance(de_view, dict) else {},
        ),
        model_features=_unique_ordered([str(item) for item in model_features]),
        target_columns=target_columns,
        prompt_path=prompt_path,
        baseline_script_path=baseline_script_path,
        baseline_error_path=baseline_error_path,
        dataset_profile_path=dataset_profile_path,
        worker_input_path=worker_input_path,
        de_context_path=de_context_path,
        quality_checks=list(case_spec.get("quality_checks") or []),
    )


def prepare_replay_workspace(case: DataEngineerReplayCase, workspace_dir: str | Path) -> Path:
    workspace = Path(workspace_dir).resolve()
    data_dir = workspace / "data"
    artifacts_dir = workspace / "artifacts"
    data_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if not case.csv_path.exists():
        raise FileNotFoundError(f"Replay CSV not found: {case.csv_path}")
    shutil.copy2(case.csv_path, data_dir / "raw.csv")

    (data_dir / "execution_contract.json").write_text(
        json.dumps(case.execution_contract, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (data_dir / "strategy_spec.json").write_text(
        json.dumps(case.strategy, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (data_dir / "de_view.json").write_text(
        json.dumps(case.de_view, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (data_dir / "required_columns.json").write_text(
        json.dumps(
            {
                "required_columns": list(case.de_view.get("required_columns") or []),
                "required_feature_selectors": list(case.de_view.get("required_feature_selectors") or []),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (data_dir / "column_sets.json").write_text(
        json.dumps(_build_column_sets_payload(case), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if case.dataset_profile_path and case.dataset_profile_path.exists():
        shutil.copy2(case.dataset_profile_path, data_dir / "dataset_profile.json")
    return workspace


def generate_script_for_case(
    case: DataEngineerReplayCase,
    *,
    primary_model: Optional[str] = None,
    fallback_model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    original_primary = os.environ.get("OPENROUTER_DE_PRIMARY_MODEL")
    original_fallback = os.environ.get("OPENROUTER_DE_FALLBACK_MODEL")
    try:
        if primary_model:
            os.environ["OPENROUTER_DE_PRIMARY_MODEL"] = primary_model
        if fallback_model:
            os.environ["OPENROUTER_DE_FALLBACK_MODEL"] = fallback_model
        agent = DataEngineerAgent(api_key=api_key)
        script = agent.generate_cleaning_script(
            data_audit=case.data_audit,
            strategy=case.strategy,
            input_path=str(case.csv_path),
            business_objective=case.business_objective,
            csv_encoding=case.csv_encoding,
            csv_sep=case.csv_sep,
            csv_decimal=case.csv_decimal,
            execution_contract=case.execution_contract,
            de_view=case.de_view,
            repair_mode=False,
        )
        return {
            "model_name": getattr(agent, "model_name", ""),
            "fallback_model_name": getattr(agent, "fallback_model_name", ""),
            "prompt": getattr(agent, "last_prompt", ""),
            "response": getattr(agent, "last_response", ""),
            "script": script,
            "preflight_issues": data_engineer_preflight(script or ""),
            "syntax_ok": _script_syntax_ok(script or ""),
        }
    finally:
        if original_primary is None:
            os.environ.pop("OPENROUTER_DE_PRIMARY_MODEL", None)
        else:
            os.environ["OPENROUTER_DE_PRIMARY_MODEL"] = original_primary
        if original_fallback is None:
            os.environ.pop("OPENROUTER_DE_FALLBACK_MODEL", None)
        else:
            os.environ["OPENROUTER_DE_FALLBACK_MODEL"] = original_fallback


def _script_syntax_ok(script: str) -> bool:
    try:
        ast.parse(str(script or ""))
        return True
    except Exception:
        return False


def execute_script_for_case(
    case: DataEngineerReplayCase,
    *,
    workspace_dir: str | Path,
    script_text: Optional[str] = None,
    script_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    workspace = prepare_replay_workspace(case, workspace_dir)
    run_script_path = workspace / "ml_script.py"
    if script_text is not None:
        run_script_path.write_text(str(script_text), encoding="utf-8")
    elif script_path:
        shutil.copy2(Path(script_path), run_script_path)
    else:
        raise ValueError("script_text or script_path is required")

    started = time.time()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    proc = subprocess.run(
        [sys.executable, str(run_script_path.name)],
        cwd=str(workspace),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        env=env,
    )
    duration = time.time() - started

    stdout_path = workspace / "stdout.txt"
    stderr_path = workspace / "stderr.txt"
    stdout_path.write_text(proc.stdout or "", encoding="utf-8")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8")

    present: List[str] = []
    missing: List[str] = []
    for rel_path in case.required_output_paths:
        output_path = workspace / rel_path
        if output_path.exists():
            present.append(rel_path)
        else:
            missing.append(rel_path)

    expected_enriched = case.expected_enriched_columns()
    enriched_path = _resolve_enriched_output_path(
        workspace,
        case.execution_contract,
        case.required_output_paths,
    )
    enriched_header = _read_csv_header(enriched_path, case.csv_sep, case.csv_encoding) if enriched_path else []
    enriched_exact_match = enriched_header == expected_enriched if expected_enriched and enriched_header else False
    enriched_subset_match = (
        bool(expected_enriched)
        and bool(enriched_header)
        and set(expected_enriched).issubset(set(enriched_header))
    )
    enriched_extra_columns = sorted(set(enriched_header) - set(expected_enriched)) if enriched_header and expected_enriched else []
    enriched_missing_columns = sorted(set(expected_enriched) - set(enriched_header)) if enriched_header and expected_enriched else list(expected_enriched)

    return {
        "returncode": int(proc.returncode),
        "duration_seconds": round(duration, 3),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "present_required_outputs": present,
        "missing_required_outputs": missing,
        "required_outputs_present_count": len(present),
        "required_outputs_missing_count": len(missing),
        "enriched_output_path": str(enriched_path) if enriched_path else "",
        "expected_enriched_columns": expected_enriched,
        "actual_enriched_columns": enriched_header,
        "enriched_schema_exact_match": bool(enriched_exact_match),
        "enriched_schema_subset_match": bool(enriched_subset_match),
        "enriched_extra_columns": enriched_extra_columns,
        "enriched_missing_columns": enriched_missing_columns,
        "success": proc.returncode == 0 and not missing,
    }


def _resolve_output_null_rate(report_payload: Dict[str, Any], column: str, *, csv_fallback_path: Optional[Path] = None) -> Optional[float]:
    """Resolve null rate for a column from report JSON, falling back to reading the CSV directly."""
    if isinstance(report_payload, dict):
        null_rates = report_payload.get("null_rates_after_cleaning")
        if isinstance(null_rates, dict) and column in null_rates:
            try:
                return float(null_rates.get(column))
            except Exception:
                pass
        column_missingness = report_payload.get("column_missingness")
        if isinstance(column_missingness, dict):
            details = column_missingness.get(column)
            if isinstance(details, dict):
                pct = details.get("missingness_pct")
                if pct is not None:
                    try:
                        value = float(pct)
                        return value / 100.0 if value > 1.0 else value
                    except Exception:
                        pass
                null_count = details.get("null_count")
                rows = (
                    report_payload.get("dataset", {}).get("output_rows_cleaned")
                    if isinstance(report_payload.get("dataset"), dict)
                    else None
                )
                try:
                    if rows:
                        return float(null_count) / float(rows)
                except Exception:
                    pass
        for section_name in (
            "missingness_after_cleaning_enriched",
            "missingness_after_cleaning_cleaned",
            "null_inflation_by_column",
        ):
            section = report_payload.get(section_name)
            if not isinstance(section, dict):
                continue
            details = section.get(column)
            if not isinstance(details, dict):
                continue
            for key in ("null_rate", "null_rate_after", "missing_rate", "missingness_rate"):
                value = details.get(key)
                if value is None:
                    continue
                try:
                    return float(value)
                except Exception:
                    pass
    # Fallback: read the output CSV directly and compute null rate
    if csv_fallback_path and csv_fallback_path.exists():
        try:
            import pandas as pd
            df = pd.read_csv(csv_fallback_path, usecols=[column], dtype=str, low_memory=False)
            total = len(df)
            if total == 0:
                return 0.0
            nulls = df[column].isna().sum() + (df[column].str.strip().str.lower().isin(["", "nan", "none", "nat"])).sum()
            return float(nulls) / float(total)
        except Exception:
            pass
    return None


def _find_enriched_csv(workspace: Path, required_output_paths: List[str]) -> Optional[Path]:
    """Find the enriched/cleaned output CSV in the workspace."""
    for rel_path in required_output_paths:
        if any(token in rel_path.lower() for token in ("enriched", "enriquec", "cleaned", "limpio", "clean_dataset")):
            candidate = workspace / rel_path
            if candidate.exists():
                return candidate
    # Glob fallback
    for pattern in (
        "artifacts/**/*enriched*.csv",
        "artifacts/**/*enriquec*.csv",
        "artifacts/**/*cleaned*.csv",
        "artifacts/**/*limpio*.csv",
        "artifacts/**/*clean*.csv",
    ):
        import glob as _glob
        matches = list(_glob.glob(str(workspace / pattern), recursive=True))
        if matches:
            return Path(matches[0])
    return None


def _run_quality_checks(case: DataEngineerReplayCase, workspace: Path) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    dataset_profile = _load_json(workspace / "data" / "dataset_profile.json", {})
    enriched_csv = _find_enriched_csv(workspace, case.required_output_paths)

    for check in case.quality_checks:
        if not isinstance(check, dict):
            continue
        ctype = str(check.get("type") or "").strip()
        if ctype == "max_null_inflation":
            report_rel = normalize_artifact_path(check.get("report_path"))
            report_payload = _load_json(workspace / report_rel, {})
            missing_frac = dataset_profile.get("missing_frac") if isinstance(dataset_profile, dict) else {}
            max_increase = float(check.get("max_increase", 0.0) or 0.0)
            columns = [str(col) for col in (check.get("columns") or []) if str(col).strip()]
            violations: List[Dict[str, Any]] = []
            for column in columns:
                try:
                    input_rate = float((missing_frac or {}).get(column, 0.0) or 0.0)
                except Exception:
                    input_rate = 0.0
                output_rate = _resolve_output_null_rate(report_payload, column, csv_fallback_path=enriched_csv)
                if output_rate is None:
                    violations.append(
                        {
                            "column": column,
                            "reason": "output_null_rate_unavailable",
                            "input_rate": input_rate,
                            "output_rate": None,
                        }
                    )
                    continue
                inflation = float(output_rate) - float(input_rate)
                if inflation > max_increase:
                    violations.append(
                        {
                            "column": column,
                            "input_rate": round(input_rate, 6),
                            "output_rate": round(float(output_rate), 6),
                            "inflation": round(inflation, 6),
                            "max_allowed_increase": max_increase,
                        }
                    )
            results.append(
                {
                    "type": ctype,
                    "passed": not violations,
                    "report_path": report_rel,
                    "violations": violations,
                }
            )
        elif ctype == "no_all_placeholder_dedup_drops":
            csv_rel = normalize_artifact_path(check.get("csv_path"))
            csv_path = workspace / csv_rel
            decision_column = str(check.get("decision_column") or "decision")
            keys_column = str(check.get("keys_column") or "dedup_keys")
            drop_token = str(check.get("drop_token") or "DROPPED").upper()
            placeholders = {
                str(token).strip().lower()
                for token in (check.get("placeholder_tokens") or [])
                if str(token).strip()
            }
            violations: List[Dict[str, Any]] = []
            if not csv_path.exists():
                violations.append({"reason": "csv_missing", "csv_path": csv_rel})
            else:
                with csv_path.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        decision = str(row.get(decision_column) or "").upper()
                        if drop_token not in decision:
                            continue
                        raw_keys = str(row.get(keys_column) or "").strip()
                        if not raw_keys:
                            continue
                        try:
                            parsed_keys = ast.literal_eval(raw_keys)
                        except Exception:
                            continue
                        if not isinstance(parsed_keys, dict) or not parsed_keys:
                            continue
                        values = [str(value or "").strip().lower() for value in parsed_keys.values()]
                        if values and all(value in placeholders for value in values):
                            violations.append(
                                {
                                    "lead_id": row.get("lead_id") or "",
                                    "decision": row.get(decision_column) or "",
                                    "dedup_keys": parsed_keys,
                                }
                            )
            results.append(
                {
                    "type": ctype,
                    "passed": not violations,
                    "csv_path": csv_rel,
                    "violations": violations,
                }
            )
    return results


def build_benchmark_summary(
    case: DataEngineerReplayCase,
    *,
    generation: Optional[Dict[str, Any]] = None,
    execution: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    generation = generation if isinstance(generation, dict) else {}
    execution = execution if isinstance(execution, dict) else {}
    script_text = str(generation.get("script") or "")
    preflight_issues = generation.get("preflight_issues")
    if not isinstance(preflight_issues, list):
        preflight_issues = data_engineer_preflight(script_text) if script_text else []
    syntax_ok = bool(generation.get("syntax_ok")) if generation else False
    if generation and "syntax_ok" not in generation:
        syntax_ok = _script_syntax_ok(script_text)
    workspace_dir = None
    if isinstance(execution, dict):
        stdout_path = str(execution.get("stdout_path") or "").strip()
        if stdout_path:
            workspace_dir = Path(stdout_path).resolve().parent
    quality_results = _run_quality_checks(case, workspace_dir or Path("."))
    quality_passed = all(bool(item.get("passed")) for item in quality_results)

    return {
        "run_id": case.run_id,
        "csv_path": str(case.csv_path),
        "required_output_paths": list(case.required_output_paths),
        "model_features": list(case.model_features),
        "target_columns": list(case.target_columns),
        "generation": {
            "model_name": str(generation.get("model_name") or ""),
            "fallback_model_name": str(generation.get("fallback_model_name") or ""),
            "syntax_ok": bool(syntax_ok),
            "preflight_issues": list(preflight_issues),
            "returned_error_prefixed_script": script_text.strip().startswith("# Error"),
            "prompt_chars": len(str(generation.get("prompt") or "")),
            "response_chars": len(str(generation.get("response") or "")),
        },
        "execution": execution,
        "quality_checks": quality_results,
        "benchmark_verdict": (
            "pass"
            if generation
            and syntax_ok
            and not preflight_issues
            and bool(execution.get("success"))
            and bool(execution.get("enriched_schema_subset_match", execution.get("enriched_schema_exact_match")))
            and quality_passed
            else "fail"
        ),
    }


def default_benchmark_output_dir(
    case: DataEngineerReplayCase,
    *,
    label: str,
) -> Path:
    return (
        case.repo_root
        / "artifacts"
        / "de_benchmarks"
        / _slug(case.run_id)
        / _slug(label)
    )
