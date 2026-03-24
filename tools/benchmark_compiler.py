"""Benchmark the Execution Planner Compiler (Task B) across multiple models.

Replays real compiler prompts from past runs against candidate models and
evaluates the resulting contracts using the existing validation pipeline.

Usage:
    python tools/benchmark_compiler.py --cases 7ae680cf e7238c3f 2cc96ce3
    python tools/benchmark_compiler.py --models openai/gpt-5.4-mini google/gemini-2.5-flash
    python tools/benchmark_compiler.py --cases 7ae680cf --models openai/gpt-5.4-mini --verbose
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from openai import OpenAI

CANDIDATE_MODELS = [
    "openai/gpt-5.4-mini",
    "openai/gpt-5.4",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    "google/gemini-3-flash-preview",
    "google/gemini-3.1-pro-preview",
    "anthropic/claude-sonnet-4-5",
    "deepseek/deepseek-chat-v3-0324",
]

OUTPUT_ROOT = REPO_ROOT / "artifacts" / "compiler_benchmarks"


# ---------------------------------------------------------------------------
# Case loading
# ---------------------------------------------------------------------------

def discover_cases() -> List[str]:
    """Find all run IDs that have compiler benchmark data."""
    runs_dir = REPO_ROOT / "runs"
    cases = []
    for run_dir in sorted(runs_dir.iterdir()):
        planner_dir = run_dir / "agents" / "execution_planner"
        if (planner_dir / "prompt_attempt_1.txt").exists() and (planner_dir / "semantic_core.json").exists():
            cases.append(run_dir.name)
    return cases


def load_case(run_id: str) -> Dict[str, Any]:
    """Load compiler benchmark inputs from a past run."""
    planner_dir = REPO_ROOT / "runs" / run_id / "agents" / "execution_planner"
    if not planner_dir.exists():
        raise FileNotFoundError(f"No planner artifacts for run {run_id}")

    prompt_path = planner_dir / "prompt_attempt_1.txt"
    semantic_core_path = planner_dir / "semantic_core.json"
    contract_full_path = planner_dir / "contract_full.json"
    contract_canonical_path = planner_dir / "contract_canonical.json"
    contract_validation_path = planner_dir / "contract_validation_report.json"

    if not prompt_path.exists():
        raise FileNotFoundError(f"No compiler prompt for run {run_id}")

    prompt = prompt_path.read_text(encoding="utf-8")
    semantic_core = json.loads(semantic_core_path.read_text(encoding="utf-8")) if semantic_core_path.exists() else {}
    original_contract = json.loads(contract_full_path.read_text(encoding="utf-8")) if contract_full_path.exists() else None
    original_canonical = json.loads(contract_canonical_path.read_text(encoding="utf-8")) if contract_canonical_path.exists() else None
    original_validation = json.loads(contract_validation_path.read_text(encoding="utf-8")) if contract_validation_path.exists() else None

    scope = semantic_core.get("scope", "unknown")

    return {
        "run_id": run_id,
        "scope": scope,
        "compiler_prompt": prompt,
        "semantic_core": semantic_core,
        "original_contract": original_contract,
        "original_canonical": original_canonical,
        "original_validation": original_validation,
    }


# ---------------------------------------------------------------------------
# Model invocation
# ---------------------------------------------------------------------------

def call_model(prompt: str, model: str, api_key: str) -> Dict[str, Any]:
    """Send the compiler prompt to a model via OpenRouter and return parsed result."""
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    t0 = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=16384,
        )
        elapsed = time.time() - t0
        raw_text = response.choices[0].message.content if response.choices else ""
        usage = response.usage
        tokens_in = getattr(usage, "prompt_tokens", 0) if usage else 0
        tokens_out = getattr(usage, "completion_tokens", 0) if usage else 0
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "elapsed_seconds": round(time.time() - t0, 2),
            "raw_text": "",
            "parsed": None,
        }

    # Parse JSON from response
    parsed, parse_error = _parse_json_response(raw_text)

    return {
        "success": parsed is not None and isinstance(parsed, dict),
        "error": str(parse_error) if parse_error else None,
        "elapsed_seconds": round(elapsed, 2),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "raw_text": raw_text,
        "parsed": parsed,
    }


def _parse_json_response(text: str):
    """Extract JSON from model response, handling markdown fences."""
    if not text or not text.strip():
        return None, "Empty response"

    clean = text.strip()
    # Strip markdown code fences
    if clean.startswith("```"):
        lines = clean.split("\n")
        # Remove first line (```json or ```)
        lines = lines[1:]
        # Remove last line if it's ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        clean = "\n".join(lines).strip()

    try:
        return json.loads(clean), None
    except json.JSONDecodeError as e:
        # Try to find JSON object in the text
        brace_start = clean.find("{")
        if brace_start >= 0:
            depth = 0
            for i in range(brace_start, len(clean)):
                if clean[i] == "{":
                    depth += 1
                elif clean[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(clean[brace_start:i+1]), None
                        except json.JSONDecodeError:
                            break
        return None, f"JSON parse error: {e}"


# ---------------------------------------------------------------------------
# Contract evaluation
# ---------------------------------------------------------------------------

def evaluate_contract(
    contract: Dict[str, Any],
    semantic_core: Dict[str, Any],
    scope: str,
) -> Dict[str, Any]:
    """Evaluate a compiled contract using the existing validation pipeline."""
    from src.utils.contract_validator import validate_contract_minimal_readonly
    from src.utils.contract_schema_registry import apply_contract_schema_registry_repairs
    from src.utils.contract_response_schema import EXECUTION_CONTRACT_CANONICAL_REQUIRED_KEYS

    # Check required keys
    required_keys = set(EXECUTION_CONTRACT_CANONICAL_REQUIRED_KEYS)
    present_keys = set(contract.keys())
    missing_keys = required_keys - present_keys
    extra_keys = present_keys - required_keys

    # Apply schema registry repairs (same as production)
    repaired = json.loads(json.dumps(contract))
    repairs = apply_contract_schema_registry_repairs(repaired)

    # Run validation
    try:
        validation = validate_contract_minimal_readonly(repaired)
    except Exception as e:
        validation = {"status": "error", "error": str(e), "issues": []}

    issues = validation.get("issues", [])
    errors = [i for i in issues if i.get("severity") == "error"]
    warnings = [i for i in issues if i.get("severity") == "warning"]

    # Specific quality checks
    quality_checks = _run_quality_checks(repaired, semantic_core, scope)

    return {
        "missing_required_keys": sorted(missing_keys),
        "extra_keys": sorted(extra_keys),
        "required_key_coverage": round((len(required_keys) - len(missing_keys)) / max(len(required_keys), 1) * 100, 1),
        "schema_repairs_applied": len(repairs) if isinstance(repairs, list) else 0,
        "validation_errors": len(errors),
        "validation_warnings": len(warnings),
        "validation_status": validation.get("status", "unknown"),
        "error_details": [{"code": e.get("code", "?"), "message": e.get("message", "?")} for e in errors[:10]],
        "warning_details": [{"code": w.get("code", "?"), "message": w.get("message", "?")} for w in warnings[:10]],
        "quality_checks": quality_checks,
        "repaired_contract": repaired,
    }


def _run_quality_checks(
    contract: Dict[str, Any],
    semantic_core: Dict[str, Any],
    scope: str,
) -> List[Dict[str, Any]]:
    """Run specific quality checks on the compiled contract."""
    checks = []

    # 1. Scope preservation
    checks.append({
        "name": "scope_preserved",
        "passed": contract.get("scope") == semantic_core.get("scope"),
        "expected": semantic_core.get("scope"),
        "actual": contract.get("scope"),
    })

    # 2. column_roles preserved
    sc_roles = semantic_core.get("column_roles", {})
    ct_roles = contract.get("column_roles", {})
    roles_match = sc_roles == ct_roles if sc_roles else True
    checks.append({
        "name": "column_roles_preserved",
        "passed": roles_match,
        "detail": "column_roles matches semantic_core" if roles_match else "column_roles diverged",
    })

    # 3. model_features preserved
    sc_features = set(semantic_core.get("model_features", []))
    ct_features = set(contract.get("model_features", []))
    features_match = sc_features == ct_features if sc_features else True
    checks.append({
        "name": "model_features_preserved",
        "passed": features_match,
        "detail": f"sc={len(sc_features)} ct={len(ct_features)}" + (
            f" missing={sc_features - ct_features}" if sc_features - ct_features else ""
        ),
    })

    # 4. manifest_path present (for cleaning scopes)
    if scope in ("cleaning_only", "full_pipeline", "data_preparation"):
        cleaned_ds = (contract.get("artifact_requirements") or {}).get("cleaned_dataset", {})
        has_manifest = bool(
            cleaned_ds.get("output_manifest_path")
            or cleaned_ds.get("manifest_path")
        )
        checks.append({
            "name": "manifest_path_present",
            "passed": has_manifest,
            "detail": cleaned_ds.get("output_manifest_path") or cleaned_ds.get("manifest_path") or "MISSING",
        })

    # 5. column_dtype_targets coverage
    canonical = contract.get("canonical_columns", [])
    dtype_targets = contract.get("column_dtype_targets", {})
    if canonical:
        covered = sum(1 for c in canonical if c in dtype_targets)
        pct = round(covered / len(canonical) * 100, 1)
        checks.append({
            "name": "dtype_coverage",
            "passed": pct >= 80,
            "detail": f"{covered}/{len(canonical)} columns ({pct}%)",
        })

    # 6. Runbook prescriptiveness check
    de_runbook = contract.get("data_engineer_runbook", "")
    ml_runbook = contract.get("ml_engineer_runbook", "")
    prescriptive_patterns = [
        r"learning_rate\s*=\s*[\d.]+",
        r"num_leaves\s*=\s*\d+",
        r"n_splits\s*=\s*\d+.*n_repeats\s*=\s*\d+",
        r"feature_fraction\s*=\s*[\d.]+",
        r"Apply median imputation",
        r"Use RepeatedStratifiedKFold",
    ]
    prescriptive_hits = []
    for pattern in prescriptive_patterns:
        for label, text in [("de_runbook", de_runbook), ("ml_runbook", ml_runbook)]:
            if isinstance(text, str) and re.search(pattern, text, re.IGNORECASE):
                prescriptive_hits.append(f"{label}: {pattern}")
    checks.append({
        "name": "runbook_not_prescriptive",
        "passed": len(prescriptive_hits) == 0,
        "detail": f"{len(prescriptive_hits)} prescriptive patterns found" + (
            f": {prescriptive_hits[:3]}" if prescriptive_hits else ""
        ),
    })

    # 7. No invented columns
    inventory_raw = semantic_core.get("canonical_columns", [])
    if not inventory_raw:
        inventory_raw = contract.get("canonical_columns", [])
    if inventory_raw:
        inventory_set = set(str(c) for c in inventory_raw)
        ct_model_features = contract.get("model_features", [])
        invented = [f for f in ct_model_features if f not in inventory_set]
        checks.append({
            "name": "no_invented_features",
            "passed": len(invented) == 0,
            "detail": f"{len(invented)} invented" + (f": {invented[:5]}" if invented else ""),
        })

    # 8. Semantic closure: model_features in required_columns
    cleaned_ds = (contract.get("artifact_requirements") or {}).get("cleaned_dataset", {})
    req_cols = set(cleaned_ds.get("required_columns", []))
    ct_model_features = set(contract.get("model_features", []))
    if ct_model_features and req_cols:
        missing_in_req = ct_model_features - req_cols
        checks.append({
            "name": "features_in_required_columns",
            "passed": len(missing_in_req) == 0,
            "detail": f"{len(missing_in_req)} model_features missing from required_columns" + (
                f": {list(missing_in_req)[:5]}" if missing_in_req else ""
            ),
        })

    # 9. Evaluation spec consistency (for ML scopes)
    if scope in ("full_pipeline", "ml_only"):
        eval_spec = contract.get("evaluation_spec", {})
        val_req = contract.get("validation_requirements", {})
        eval_metric = eval_spec.get("primary_metric", "")
        val_metric = val_req.get("primary_metric", "")
        metrics_consistent = eval_metric == val_metric if eval_metric and val_metric else True
        checks.append({
            "name": "metric_consistency",
            "passed": metrics_consistent and bool(eval_metric),
            "detail": f"eval={eval_metric} val={val_metric}",
        })

    # 10. Gate quality
    for gate_key in ("cleaning_gates", "qa_gates"):
        gates = contract.get(gate_key, [])
        if isinstance(gates, list) and gates:
            has_hard = any(g.get("severity") == "HARD" for g in gates if isinstance(g, dict))
            all_named = all(isinstance(g, dict) and g.get("name") for g in gates)
            checks.append({
                "name": f"{gate_key}_quality",
                "passed": has_hard and all_named,
                "detail": f"{len(gates)} gates, has_hard={has_hard}, all_named={all_named}",
            })

    return checks


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_result(evaluation: Dict[str, Any]) -> Dict[str, Any]:
    """Compute a summary score from evaluation results."""
    checks = evaluation.get("quality_checks", [])
    passed = sum(1 for c in checks if c.get("passed"))
    total = len(checks)

    # Weighted score
    key_coverage = evaluation.get("required_key_coverage", 0)
    val_errors = evaluation.get("validation_errors", 99)
    val_warnings = evaluation.get("validation_warnings", 99)
    quality_pct = round(passed / max(total, 1) * 100, 1)

    # Composite: key_coverage(30%) + quality_checks(40%) + validation_penalty(30%)
    val_penalty = max(0, 100 - val_errors * 15 - val_warnings * 3)
    composite = round(key_coverage * 0.30 + quality_pct * 0.40 + val_penalty * 0.30, 1)

    return {
        "composite_score": composite,
        "key_coverage_pct": key_coverage,
        "quality_checks_passed": f"{passed}/{total}",
        "quality_pct": quality_pct,
        "validation_errors": val_errors,
        "validation_warnings": val_warnings,
        "validation_penalty_score": val_penalty,
    }


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------

def run_single(run_id: str, model: str, api_key: str, verbose: bool = False) -> Dict[str, Any]:
    """Run a single compiler benchmark: model x case."""
    print(f"\n{'='*70}")
    print(f"  COMPILER BENCHMARK: run={run_id} | model={model}")
    print(f"{'='*70}")

    try:
        case = load_case(run_id)
    except FileNotFoundError as e:
        print(f"  SKIP: {e}")
        return {"run_id": run_id, "model": model, "verdict": "skip", "error": str(e)}

    print(f"  Scope: {case['scope']} | Prompt size: {len(case['compiler_prompt']):,} chars")

    # Call model
    print(f"  Calling {model}...")
    result = call_model(case["compiler_prompt"], model, api_key)

    if not result["success"]:
        print(f"  FAILED: {result.get('error', 'unknown')}")
        # Save artifacts
        out_dir = OUTPUT_ROOT / run_id / _slug(model)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "raw_response.txt").write_text(result.get("raw_text", ""), encoding="utf-8")
        (out_dir / "result.json").write_text(json.dumps({
            "run_id": run_id, "model": model, "verdict": "parse_failure",
            "error": result.get("error"), "elapsed_seconds": result.get("elapsed_seconds"),
            "tokens_in": result.get("tokens_in", 0), "tokens_out": result.get("tokens_out", 0),
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        return {
            "run_id": run_id, "model": model, "verdict": "parse_failure",
            "error": result.get("error"),
            "elapsed_seconds": result.get("elapsed_seconds"),
        }

    print(f"  Response: {result['elapsed_seconds']}s | {result['tokens_in']} in / {result['tokens_out']} out")

    # Evaluate
    contract = result["parsed"]
    evaluation = evaluate_contract(contract, case["semantic_core"], case["scope"])
    scores = score_result(evaluation)

    print(f"  Key coverage: {evaluation['required_key_coverage']}%")
    print(f"  Validation: {evaluation['validation_errors']} errors, {evaluation['validation_warnings']} warnings")
    print(f"  Schema repairs: {evaluation['schema_repairs_applied']}")

    quality_checks = evaluation.get("quality_checks", [])
    failed_checks = [c for c in quality_checks if not c.get("passed")]
    passed_checks = [c for c in quality_checks if c.get("passed")]
    print(f"  Quality: {len(passed_checks)}/{len(quality_checks)} checks passed")
    for c in failed_checks:
        print(f"    FAIL: {c['name']} — {c.get('detail', '')}")

    print(f"  Composite score: {scores['composite_score']}")

    if verbose:
        for c in passed_checks:
            print(f"    PASS: {c['name']} — {c.get('detail', '')}")
        if evaluation.get("error_details"):
            print(f"  Validation errors:")
            for e in evaluation["error_details"][:5]:
                print(f"    {e['code']}: {e['message'][:100]}")

    # Save artifacts
    out_dir = OUTPUT_ROOT / run_id / _slug(model)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raw_response.txt").write_text(result.get("raw_text", ""), encoding="utf-8")
    (out_dir / "contract.json").write_text(json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    summary = {
        "run_id": run_id,
        "model": model,
        "scope": case["scope"],
        "verdict": "pass" if scores["composite_score"] >= 60 and evaluation["validation_errors"] <= 2 else "fail",
        "elapsed_seconds": result.get("elapsed_seconds"),
        "tokens_in": result.get("tokens_in", 0),
        "tokens_out": result.get("tokens_out", 0),
        "scores": scores,
        "missing_required_keys": evaluation["missing_required_keys"],
        "validation_errors": evaluation["validation_errors"],
        "validation_warnings": evaluation["validation_warnings"],
        "schema_repairs": evaluation["schema_repairs_applied"],
        "quality_checks_summary": {c["name"]: c["passed"] for c in quality_checks},
        "failed_checks": [{k: v for k, v in c.items() if k != "passed"} for c in failed_checks],
    }
    (out_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"  VERDICT: {summary['verdict'].upper()}")
    return summary


def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip()).strip("._-")
    return text or "default"


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def print_summary_table(results: List[Dict[str, Any]]) -> None:
    """Print a readable results table."""
    print(f"\n\n{'='*100}")
    print(f"  COMPILER BENCHMARK RESULTS")
    print(f"{'='*100}")
    print(f"{'Model':<38} {'Case':<10} {'Scope':<16} {'Score':>6} {'Keys':>5} {'Errs':>5} {'Warns':>5} {'Quality':>8} {'Verdict':<8}")
    print(f"{'-'*38} {'-'*10} {'-'*16} {'-'*6} {'-'*5} {'-'*5} {'-'*5} {'-'*8} {'-'*8}")

    for r in results:
        model = str(r.get("model", "?"))[:37]
        run_id = str(r.get("run_id", "?"))[:9]
        scope = str(r.get("scope", "?"))[:15]
        scores = r.get("scores", {})
        composite = scores.get("composite_score", 0)
        key_cov = f"{scores.get('key_coverage_pct', 0):.0f}%"
        errs = str(r.get("validation_errors", "?"))
        warns = str(r.get("validation_warnings", "?"))
        quality = scores.get("quality_checks_passed", "?")
        verdict = str(r.get("verdict", "?")).upper()
        print(f"{model:<38} {run_id:<10} {scope:<16} {composite:>6.1f} {key_cov:>5} {errs:>5} {warns:>5} {quality:>8} {verdict:<8}")

    # Model averages
    print(f"\n{'='*70}")
    print(f"  MODEL AVERAGES")
    print(f"{'='*70}")
    model_data: Dict[str, List[float]] = {}
    model_verdicts: Dict[str, Dict[str, int]] = {}
    for r in results:
        m = r.get("model", "?")
        s = r.get("scores", {}).get("composite_score", 0)
        v = r.get("verdict", "fail")
        model_data.setdefault(m, []).append(s)
        model_verdicts.setdefault(m, {"pass": 0, "fail": 0, "total": 0})
        model_verdicts[m]["total"] += 1
        model_verdicts[m]["pass" if v == "pass" else "fail"] += 1

    rows = []
    for m, scores_list in model_data.items():
        avg = sum(scores_list) / len(scores_list)
        v = model_verdicts[m]
        rows.append((m, avg, v["pass"], v["total"]))

    for m, avg, p, t in sorted(rows, key=lambda x: -x[1]):
        bar = "#" * int(avg / 2)
        print(f"  {m:<38} avg={avg:5.1f}  pass={p}/{t}  {bar}")


def main():
    parser = argparse.ArgumentParser(description="Compiler Task B benchmark across models")
    parser.add_argument("--models", nargs="*", default=CANDIDATE_MODELS, help="Models to test")
    parser.add_argument("--cases", nargs="*", default=None, help="Run IDs to test (default: auto-select diverse set)")
    parser.add_argument("--api-key", default="", help="OpenRouter API key override")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output per check")
    parser.add_argument("--list-cases", action="store_true", help="List available cases and exit")
    args = parser.parse_args()

    if args.list_cases:
        cases = discover_cases()
        for c in cases:
            case = load_case(c)
            print(f"  {c}  scope={case['scope']:<20}  prompt_size={len(case['compiler_prompt']):>6,} chars")
        print(f"\n  Total: {len(cases)} cases")
        return 0

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: No API key. Set OPENROUTER_API_KEY or use --api-key")
        return 1

    # Select cases: prefer diversity of scopes
    if args.cases:
        case_ids = args.cases
    else:
        # Auto-select: 1 full_pipeline + 1 cleaning_only + 1 data_preparation if available
        all_cases = discover_cases()
        by_scope: Dict[str, List[str]] = {}
        for c in all_cases:
            case = load_case(c)
            by_scope.setdefault(case["scope"], []).append(c)

        case_ids = []
        for preferred_scope in ["full_pipeline", "cleaning_only", "data_preparation"]:
            if preferred_scope in by_scope:
                case_ids.append(by_scope[preferred_scope][0])
        if not case_ids:
            case_ids = all_cases[:3]

    print(f"Models: {args.models}")
    print(f"Cases:  {case_ids}")
    print(f"Total:  {len(args.models) * len(case_ids)} benchmarks")

    results = []
    for model in args.models:
        for run_id in case_ids:
            summary = run_single(run_id, model, api_key, verbose=args.verbose)
            results.append(summary)

    print_summary_table(results)

    # Save batch results
    batch_path = OUTPUT_ROOT / "batch_results.json"
    batch_path.parent.mkdir(parents=True, exist_ok=True)
    batch_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFull results: {batch_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
