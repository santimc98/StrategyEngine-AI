"""Batch runner for Data Engineer replay benchmarks across multiple models and cases."""
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.data_engineer_replay_benchmark import (
    build_benchmark_summary,
    default_benchmark_output_dir,
    execute_script_for_case,
    generate_script_for_case,
    load_data_engineer_replay_case,
)

# Models to benchmark via OpenRouter
CANDIDATE_MODELS = [
    "openai/gpt-5.4-mini",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    "anthropic/claude-sonnet-4-5",
    "deepseek/deepseek-r1",
]

BENCHMARK_CASES = ["8ec99856", "ae947942", "c946b64d"]


def _slug(value: str) -> str:
    import re
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip()).strip("._-")
    return text or "default"


def run_single_benchmark(run_id: str, model: str, api_key: str = "") -> dict:
    """Run a single benchmark case with a specific model. Returns summary dict."""
    print(f"\n{'='*70}")
    print(f"  BENCHMARK: run={run_id} | model={model}")
    print(f"{'='*70}")

    try:
        case = load_data_engineer_replay_case(run_id)
    except FileNotFoundError as e:
        print(f"  SKIP: {e}")
        return {"run_id": run_id, "model": model, "benchmark_verdict": "skip", "error": str(e)}

    output_dir = default_benchmark_output_dir(case, label=model)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate
    t0 = time.time()
    print(f"  Generating script with {model}...")
    try:
        generation = generate_script_for_case(
            case,
            primary_model=model,
            api_key=api_key or None,
        )
        gen_time = time.time() - t0
        print(f"  Generation done in {gen_time:.1f}s | syntax_ok={generation.get('syntax_ok')} | preflight={len(generation.get('preflight_issues', []))} issues")
    except Exception as e:
        print(f"  Generation FAILED: {e}")
        return {"run_id": run_id, "model": model, "benchmark_verdict": "generation_error", "error": str(e)}

    # Save generation artifacts
    (output_dir / "generated_prompt.txt").write_text(str(generation.get("prompt") or ""), encoding="utf-8")
    (output_dir / "generated_response.txt").write_text(str(generation.get("response") or ""), encoding="utf-8")
    (output_dir / "generated_script.py").write_text(str(generation.get("script") or ""), encoding="utf-8")

    # Execute
    print(f"  Executing generated script...")
    t0 = time.time()
    try:
        execution = execute_script_for_case(
            case,
            workspace_dir=output_dir,
            script_text=str(generation.get("script") or ""),
        )
        exec_time = time.time() - t0
        print(f"  Execution done in {exec_time:.1f}s | returncode={execution.get('returncode')} | outputs={execution.get('required_outputs_present_count')}/{execution.get('required_outputs_present_count',0)+execution.get('required_outputs_missing_count',0)}")
        print(f"  Schema subset match: {execution.get('enriched_schema_subset_match')} | exact: {execution.get('enriched_schema_exact_match')}")
        if execution.get("enriched_extra_columns"):
            print(f"  Extra columns: {execution.get('enriched_extra_columns')}")
        if execution.get("enriched_missing_columns"):
            print(f"  Missing columns: {execution.get('enriched_missing_columns')}")
    except Exception as e:
        print(f"  Execution FAILED: {e}")
        execution = {"success": False, "error": str(e)}

    # Summary
    summary = build_benchmark_summary(case, generation=generation, execution=execution)
    summary["generation_time_seconds"] = round(gen_time, 1)
    summary["model"] = model
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    verdict = summary.get("benchmark_verdict", "?")
    print(f"  VERDICT: {verdict.upper()}")
    for qc in summary.get("quality_checks", []):
        print(f"  Quality check [{qc.get('type')}]: passed={qc.get('passed')} violations={len(qc.get('violations', []))}")

    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch benchmark runner")
    parser.add_argument("--models", nargs="*", default=CANDIDATE_MODELS, help="Models to test")
    parser.add_argument("--cases", nargs="*", default=BENCHMARK_CASES, help="Run IDs to test")
    parser.add_argument("--api-key", default="", help="OpenRouter API key override")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "")
    results = []

    for model in args.models:
        for run_id in args.cases:
            summary = run_single_benchmark(run_id, model, api_key=api_key)
            results.append(summary)

    # Print summary table
    print(f"\n\n{'='*90}")
    print(f"  BENCHMARK RESULTS SUMMARY")
    print(f"{'='*90}")
    print(f"{'Model':<35} {'Case':<12} {'Verdict':<10} {'Outputs':<10} {'Schema':<10} {'Quality':<10}")
    print(f"{'-'*35} {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for r in results:
        model = str(r.get("model", "?"))[:34]
        run_id = str(r.get("run_id", "?"))[:11]
        verdict = str(r.get("benchmark_verdict", "?"))
        exe = r.get("execution", {})
        outputs = f"{exe.get('required_outputs_present_count', '?')}/{exe.get('required_outputs_present_count', 0) + exe.get('required_outputs_missing_count', 0)}" if isinstance(exe, dict) else "?"
        schema = "subset" if exe.get("enriched_schema_subset_match") else ("exact" if exe.get("enriched_schema_exact_match") else "no")
        qcs = r.get("quality_checks", [])
        quality = "N/A" if not qcs else ("OK" if all(q.get("passed") for q in qcs) else "FAIL")
        print(f"{model:<35} {run_id:<12} {verdict:<10} {outputs:<10} {schema:<10} {quality:<10}")

    # Save full results
    output_path = REPO_ROOT / "artifacts" / "de_benchmarks" / "batch_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFull results saved to: {output_path}")

    # Score summary per model
    print(f"\n{'='*60}")
    print(f"  MODEL SCORES")
    print(f"{'='*60}")
    model_scores = {}
    for r in results:
        model = r.get("model", "?")
        if model not in model_scores:
            model_scores[model] = {"pass": 0, "fail": 0, "total": 0}
        model_scores[model]["total"] += 1
        if r.get("benchmark_verdict") == "pass":
            model_scores[model]["pass"] += 1
        else:
            model_scores[model]["fail"] += 1

    for model, scores in sorted(model_scores.items(), key=lambda x: -x[1]["pass"]):
        pct = (scores["pass"] / scores["total"] * 100) if scores["total"] else 0
        print(f"  {model:<35} {scores['pass']}/{scores['total']} passed ({pct:.0f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
