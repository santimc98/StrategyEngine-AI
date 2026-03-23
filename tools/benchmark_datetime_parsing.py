"""
Focused benchmark: datetime parsing quality on mixed-format CRM data.

Tests whether a model can correctly parse datetime columns with mixed formats
(ISO, DD-MM-YYYY, slash dates, placeholders, impossible dates) WITHOUT
inflating null rates beyond acceptable thresholds.

Uses the real run 8fa4fbed context and data.
"""
import json
import os
import sys
import time
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODELS = [
    "openai/gpt-5.4-mini",
    "openai/gpt-5.4-nano",
    "openai/gpt-5.4",
    "anthropic/claude-opus-4-6",
]

# The real CSV
CSV_PATH = REPO_ROOT / "data" / "synthetic_dirty_crm_leads.csv"

# Datetime columns to evaluate
DT_COLUMNS = ["created_at", "last_activity_at", "next_followup_at"]

# Maximum acceptable null inflation (percentage points) over raw null rate
MAX_NULL_INFLATION_PP = 35.0

# The focused prompt: just the datetime parsing task with full context
SYSTEM_PROMPT = """
You are a Senior Data Engineer. Your ONLY task is to write a Python script that
parses datetime columns from a dirty CRM CSV, handling mixed formats correctly.

=== TASK ===
Read the CSV at 'data/raw.csv' (dtype=str, encoding=utf-8, sep=',').
Parse ONLY these datetime columns: {dt_columns}
Write the result to 'output/parsed_dates.csv' with the same rows and columns as input,
but with the datetime columns parsed to proper datetime types.
Also write 'output/parsing_report.json' with null rates before and after parsing per column.

=== DATASET PROFILE (from real data) ===
{dataset_profile}

=== CRITICAL REQUIREMENTS ===
1. The dataset mixes MULTIPLE datetime formats in the same column:
   - ISO: "2025-07-04", "2024-11-22"
   - DD-MM-YYYY: "04-12-2025", "15-03-2025"
   - Slash dates: "01/11/2024", "03/07/2025"
   - Impossible dates: "2025-13-40" (month=13, day=40) — must become NaT
   - Placeholders: "not_a_date" — must become NaT
   - Real NaN (missing values) — must remain NaT

2. You MUST use a MULTI-STAGE parsing strategy:
   - A single pd.to_datetime() call with dayfirst=True will DESTROY ISO dates
     (e.g., "2025-07-04" interpreted as day=20, month=25 → NaT)
   - A single call with dayfirst=False will DESTROY DD-MM-YYYY dates
   - You need at least 2 parsing passes or a format-detection approach

3. NULL INFLATION GUARDRAIL:
   - created_at raw null rate: ~24.2%. After parsing, must stay below ~59% (24.2 + 35pp max).
   - last_activity_at raw null rate: ~22.5%. After parsing, must stay below ~57.5%.
   - The impossible dates ("2025-13-40") and placeholders ("not_a_date") WILL become NaT.
     That's ~29% additional nulls. So expected final null rate is ~53-55%, NOT 94%.
   - If your script produces >60% nulls on created_at, YOUR PARSER IS BROKEN.

4. PANDAS COMPATIBILITY (pandas 2.x):
   - Do NOT use infer_datetime_format=True (removed in pandas 2.x)
   - Do NOT use datetime_is_numeric=True in describe()
   - print() must use ASCII-safe characters only (no unicode checkmarks/arrows)

=== OUTPUT FORMAT ===
Return ONLY valid Python code. No markdown fences, no prose.
The script must:
- Read from 'data/raw.csv'
- Write parsed CSV to 'output/parsed_dates.csv'
- Write JSON report to 'output/parsing_report.json' with structure:
  {{"columns": {{"col_name": {{"raw_null_pct": float, "parsed_null_pct": float, "inflation_pp": float}}}}}}
"""


def _build_profile_for_dt_columns(csv_path: Path) -> str:
    """Build a focused profile for datetime columns."""
    import pandas as pd
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    profiles = {}
    for col in DT_COLUMNS:
        if col not in df.columns:
            continue
        series = df[col]
        null_count = int(series.isna().sum())
        non_null = series.dropna()
        total = len(df)
        vc = non_null.value_counts().head(8)
        # Detect format patterns
        format_sigs = set()
        for val in non_null.head(50).tolist():
            sig = re.sub(r"\d", "D", str(val).strip())
            format_sigs.add(sig)
        profiles[col] = {
            "total_rows": total,
            "null_count": null_count,
            "null_pct": round(null_count / total * 100, 1),
            "unique_count": int(non_null.nunique()),
            "top_values": {str(k): int(v) for k, v in vc.items()},
            "observed_format_patterns": sorted(format_sigs)[:8],
        }
    return json.dumps(profiles, indent=2, ensure_ascii=False)


def _extract_code(response: str) -> str:
    """Extract Python code from response, stripping markdown fences."""
    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        start = 1
        end = len(lines)
        for i in range(1, len(lines)):
            if lines[i].strip() == "```":
                end = i
                break
        text = "\n".join(lines[start:end])
    return text.strip()


def run_benchmark(model: str, api_key: str) -> dict:
    """Run the datetime parsing benchmark for a single model."""
    import pandas as pd
    from openai import OpenAI
    import subprocess
    import tempfile
    import shutil

    print(f"\n{'='*70}")
    print(f"  MODEL: {model}")
    print(f"{'='*70}")

    profile = _build_profile_for_dt_columns(CSV_PATH)
    prompt = SYSTEM_PROMPT.format(
        dt_columns=json.dumps(DT_COLUMNS),
        dataset_profile=profile,
    )

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        timeout=300.0,
    )

    # Generate
    t0 = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Write the datetime parsing script. Return only Python code."},
            ],
            temperature=0.0,
        )
        gen_time = round(time.time() - t0, 1)
        raw_response = response.choices[0].message.content or ""
        script = _extract_code(raw_response)
        print(f"  Generation: {gen_time}s | {len(script)} chars")
    except Exception as e:
        print(f"  Generation FAILED: {e}")
        return {"model": model, "verdict": "generation_error", "error": str(e)}

    # Setup workspace
    workspace = Path(tempfile.mkdtemp(prefix=f"dt_bench_{model.replace('/', '_')}_"))
    (workspace / "data").mkdir()
    (workspace / "output").mkdir()
    shutil.copy2(CSV_PATH, workspace / "data" / "raw.csv")
    script_path = workspace / "parse_dates.py"
    script_path.write_text(script, encoding="utf-8")

    # Save script for inspection
    artifacts_dir = REPO_ROOT / "artifacts" / "dt_benchmarks" / model.replace("/", "_")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "script.py").write_text(script, encoding="utf-8")
    (artifacts_dir / "response.txt").write_text(raw_response, encoding="utf-8")

    # Execute
    t0 = time.time()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    proc = subprocess.run(
        [sys.executable, "parse_dates.py"],
        cwd=str(workspace),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        env=env,
        timeout=60,
    )
    exec_time = round(time.time() - t0, 1)

    if proc.returncode != 0:
        print(f"  Execution FAILED (rc={proc.returncode})")
        stderr_tail = proc.stderr[-500:] if proc.stderr else ""
        print(f"  stderr: {stderr_tail}")
        (artifacts_dir / "stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
        shutil.rmtree(workspace, ignore_errors=True)
        return {
            "model": model,
            "verdict": "runtime_error",
            "gen_time": gen_time,
            "exec_time": exec_time,
            "error": stderr_tail,
        }

    # Analyze results
    parsed_csv = workspace / "output" / "parsed_dates.csv"
    report_json = workspace / "output" / "parsing_report.json"

    result = {
        "model": model,
        "gen_time": gen_time,
        "exec_time": exec_time,
        "columns": {},
    }

    # Read raw for comparison
    raw_df = pd.read_csv(CSV_PATH, dtype=str, low_memory=False)

    if parsed_csv.exists():
        parsed_df = pd.read_csv(parsed_csv, low_memory=False)
        print(f"  Execution OK ({exec_time}s) | parsed shape: {parsed_df.shape}")

        for col in DT_COLUMNS:
            raw_null_pct = round(raw_df[col].isna().sum() / len(raw_df) * 100, 1) if col in raw_df.columns else 0
            if col in parsed_df.columns:
                parsed_null_pct = round(parsed_df[col].isna().sum() / len(parsed_df) * 100, 1)
                inflation = round(parsed_null_pct - raw_null_pct, 1)
                passed = bool(inflation <= MAX_NULL_INFLATION_PP)
                result["columns"][col] = {
                    "raw_null_pct": float(raw_null_pct),
                    "parsed_null_pct": float(parsed_null_pct),
                    "inflation_pp": float(inflation),
                    "passed": passed,
                }
                status = "OK" if passed else "FAIL"
                print(f"  {col}: {raw_null_pct}% -> {parsed_null_pct}% (inflation={inflation:+.1f}pp) [{status}]")
            else:
                result["columns"][col] = {"error": "column_missing_in_output"}
                print(f"  {col}: MISSING from output")
    else:
        print(f"  Output CSV not found at {parsed_csv}")
        result["verdict"] = "no_output"
        shutil.rmtree(workspace, ignore_errors=True)
        return result

    # Save model report if it wrote one
    if report_json.exists():
        shutil.copy2(report_json, artifacts_dir / "parsing_report.json")

    # Copy parsed CSV for inspection
    if parsed_csv.exists():
        shutil.copy2(parsed_csv, artifacts_dir / "parsed_dates.csv")

    # Verdict
    all_passed = all(
        c.get("passed", False)
        for c in result["columns"].values()
        if "error" not in c
    )
    result["verdict"] = "pass" if all_passed and result["columns"] else "fail"
    print(f"  VERDICT: {result['verdict'].upper()}")

    shutil.rmtree(workspace, ignore_errors=True)

    # Save summary
    (artifacts_dir / "benchmark_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Datetime parsing benchmark")
    parser.add_argument("--models", nargs="*", default=MODELS)
    parser.add_argument("--api-key", default="")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY required")
        return 1

    results = []
    for model in args.models:
        r = run_benchmark(model, api_key)
        results.append(r)

    # Summary table
    print(f"\n\n{'='*90}")
    print(f"  DATETIME PARSING BENCHMARK — RESULTS")
    print(f"{'='*90}")
    print(f"{'Model':<30} {'Verdict':<10} {'GenTime':<8} ", end="")
    for col in DT_COLUMNS:
        print(f"{col:<22} ", end="")
    print()
    print("-" * 120)

    for r in results:
        model = r.get("model", "?")[:29]
        verdict = r.get("verdict", "?")
        gen = f"{r.get('gen_time', '?')}s"
        print(f"{model:<30} {verdict:<10} {gen:<8} ", end="")
        for col in DT_COLUMNS:
            cd = r.get("columns", {}).get(col, {})
            if "error" in cd:
                print(f"{'ERROR':<22} ", end="")
            elif "parsed_null_pct" in cd:
                info = f"{cd['raw_null_pct']}%->{cd['parsed_null_pct']}% ({cd['inflation_pp']:+.0f}pp)"
                tag = " OK" if cd.get("passed") else " FAIL"
                print(f"{info+tag:<22} ", end="")
            else:
                print(f"{'N/A':<22} ", end="")
        print()

    # Save all results
    out_path = REPO_ROOT / "artifacts" / "dt_benchmarks" / "all_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
