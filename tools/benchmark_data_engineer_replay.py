import argparse
import json
import sys
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
from src.utils.data_engineer_preflight import data_engineer_preflight


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the Data Engineer agent on a replayed real run without executing the full pipeline.",
    )
    parser.add_argument("--run-id", required=True, help="Run id to replay, e.g. 8ec99856")
    parser.add_argument(
        "--mode",
        choices=["baseline", "generate", "script"],
        default="baseline",
        help="baseline=execute historical DE script; generate=generate current DE script with selected model and execute it; script=execute a provided script path",
    )
    parser.add_argument("--model", default="", help="Primary DE model override for --mode generate")
    parser.add_argument("--fallback-model", default="", help="Fallback DE model override for --mode generate")
    parser.add_argument("--script-path", default="", help="Script path for --mode script")
    parser.add_argument("--api-key", default="", help="Optional OpenRouter API key override for generation")
    parser.add_argument("--output-dir", default="", help="Directory where benchmark artifacts will be written")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    case = load_data_engineer_replay_case(args.run_id)

    if args.mode == "baseline":
        if not case.baseline_script_path or not case.baseline_script_path.exists():
            print("Baseline script not found for this run.")
            return 1
        label = "baseline"
    elif args.mode == "generate":
        label = args.model or "current_configured_model"
    else:
        if not args.script_path:
            print("--script-path is required when --mode script")
            return 1
        label = Path(args.script_path).stem

    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_benchmark_output_dir(case, label=label)
    output_dir.mkdir(parents=True, exist_ok=True)

    generation = {}
    if args.mode == "generate":
        generation = generate_script_for_case(
            case,
            primary_model=str(args.model or "").strip() or None,
            fallback_model=str(args.fallback_model or "").strip() or None,
            api_key=str(args.api_key or "").strip() or None,
        )
        (output_dir / "generated_prompt.txt").write_text(
            str(generation.get("prompt") or ""),
            encoding="utf-8",
        )
        (output_dir / "generated_response.txt").write_text(
            str(generation.get("response") or ""),
            encoding="utf-8",
        )
        (output_dir / "generated_script.py").write_text(
            str(generation.get("script") or ""),
            encoding="utf-8",
        )
    elif args.mode == "baseline":
        script_text = case.baseline_script_path.read_text(encoding="utf-8") if case.baseline_script_path else ""
        generation = {
            "model_name": "baseline_replay",
            "fallback_model_name": "",
            "prompt": case.prompt_path.read_text(encoding="utf-8") if case.prompt_path and case.prompt_path.exists() else "",
            "response": "",
            "script": script_text,
            "preflight_issues": data_engineer_preflight(script_text),
            "syntax_ok": True,
        }
    else:
        script_text = Path(args.script_path).read_text(encoding="utf-8")
        generation = {
            "model_name": "external_script",
            "fallback_model_name": "",
            "prompt": "",
            "response": "",
            "script": script_text,
            "preflight_issues": data_engineer_preflight(script_text),
            "syntax_ok": True,
        }

    case_manifest = {
        "run_id": case.run_id,
        "csv_path": str(case.csv_path),
        "prompt_path": str(case.prompt_path) if case.prompt_path else "",
        "baseline_script_path": str(case.baseline_script_path) if case.baseline_script_path else "",
        "baseline_error_path": str(case.baseline_error_path) if case.baseline_error_path else "",
        "required_output_paths": case.required_output_paths,
        "model_features": case.model_features,
        "target_columns": case.target_columns,
    }
    (output_dir / "case_manifest.json").write_text(
        json.dumps(case_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if args.mode == "baseline":
        execution = execute_script_for_case(case, workspace_dir=output_dir, script_path=case.baseline_script_path)
    elif args.mode == "script":
        execution = execute_script_for_case(case, workspace_dir=output_dir, script_path=args.script_path)
    else:
        execution = execute_script_for_case(
            case,
            workspace_dir=output_dir,
            script_text=str(generation.get("script") or ""),
        )

    summary = build_benchmark_summary(case, generation=generation, execution=execution)
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nBenchmark artifacts written to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
