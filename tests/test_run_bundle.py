import json
import os
import time
from pathlib import Path

from src.utils.run_bundle import init_run_bundle, write_run_manifest, copy_run_artifacts, log_agent_snapshot
from src.utils.run_logger import init_run_log, log_run_event


def test_run_bundle_creates_manifest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv_path = tmp_path / "input.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

    run_id = "run1234"
    state = {
        "run_id": run_id,
        "run_start_ts": "2025-01-01T00:00:00",
        "csv_path": str(csv_path),
        "csv_encoding": "utf-8",
        "csv_sep": ",",
        "csv_decimal": ".",
        "agent_models": {"steward": "test-model"},
    }
    run_dir = init_run_bundle(run_id, state, base_dir=str(tmp_path / "runs"), enable_tee=False)
    contracts_dir = Path(run_dir) / "contracts"
    artifacts_dir = Path(run_dir) / "artifacts" / "data"
    contracts_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    contract = {"required_outputs": ["data/metrics.json"]}
    (contracts_dir / "execution_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    (artifacts_dir / "metrics.json").write_text("{}", encoding="utf-8")
    manifest_path = write_run_manifest(run_id, state)
    assert manifest_path is not None
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    assert manifest["run_id"] == run_id
    assert manifest["input"]["path"] == str(csv_path)
    assert "data/metrics.json" in manifest["required_outputs"]
    assert "data/metrics.json" in manifest["produced_outputs"]


def test_events_jsonl_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_id = "run5678"
    run_dir = init_run_bundle(run_id, {}, base_dir=str(tmp_path / "runs"), enable_tee=False)
    init_run_log(run_id, {"note": "test"})
    log_run_event(run_id, "test_event", {"ok": True})
    events_path = Path(run_dir) / "events.jsonl"
    assert events_path.exists()
    content = events_path.read_text(encoding="utf-8")
    assert "test_event" in content


def test_manifest_no_ml_outputs_without_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_id = "run9999"
    csv_path = tmp_path / "input.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")
    state = {
        "run_id": run_id,
        "run_start_ts": "2025-01-01T00:00:00",
        "csv_path": str(csv_path),
        "csv_encoding": "utf-8",
        "csv_sep": ",",
        "csv_decimal": ".",
    }
    run_dir = init_run_bundle(run_id, state, base_dir=str(tmp_path / "runs"), enable_tee=False)
    contracts_dir = Path(run_dir) / "contracts"
    artifacts_dir = Path(run_dir) / "artifacts" / "data"
    contracts_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    contract = {"required_outputs": ["data/cleaned_data.csv", "data/metrics.json"]}
    (contracts_dir / "execution_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    (artifacts_dir / "cleaned_data.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    manifest_path = write_run_manifest(run_id, state, status_final="FAIL")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    assert "data/metrics.json" not in manifest["produced_outputs"]


def test_copy_run_artifacts_filters_by_mtime(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_id = "run2222"
    run_dir = init_run_bundle(run_id, {}, base_dir=str(tmp_path / "runs"), enable_tee=False)
    source_dir = tmp_path / "data"
    source_dir.mkdir(parents=True, exist_ok=True)
    old_file = source_dir / "old.txt"
    new_file = source_dir / "new.txt"
    old_file.write_text("old", encoding="utf-8")
    new_file.write_text("new", encoding="utf-8")

    now = time.time()
    os.utime(old_file, (now - 10, now - 10))
    os.utime(new_file, (now + 2, now + 2))

    copy_run_artifacts(run_id, [str(source_dir)], since_epoch=now - 1)

    dest_old = Path(run_dir) / "artifacts" / "data" / "old.txt"
    dest_new = Path(run_dir) / "artifacts" / "data" / "new.txt"
    assert not dest_old.exists()
    assert dest_new.exists()


def test_run_manifest_includes_iteration_trace_metadata(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_id = "run_trace_01"
    state = {
        "run_id": run_id,
        "run_start_ts": "2026-02-20T00:00:00",
        "ml_improvement_round_count": 1,
        "ml_improvement_attempted": True,
        "ml_improvement_kept": "baseline",
    }
    run_dir = init_run_bundle(run_id, state, base_dir=str(tmp_path / "runs"), enable_tee=False)

    governance_dir = Path(run_dir) / "report" / "governance"
    governance_dir.mkdir(parents=True, exist_ok=True)
    journal_path = governance_dir / "ml_iteration_journal.jsonl"
    journal_path.write_text(
        json.dumps(
            {
                "iteration_id": 2,
                "stage": "review_complete",
                "reviewer_verdict": "NEEDS_IMPROVEMENT",
                "qa_verdict": "APPROVE_WITH_WARNINGS",
                "outputs_missing": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    summary_path = governance_dir / "ml_iteration_trace_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "entries_count": 1,
                "stages_count": {"review_complete": 1},
                "last_entry": {"iteration_id": 2, "stage": "review_complete"},
            }
        ),
        encoding="utf-8",
    )

    manifest_path = write_run_manifest(run_id, state)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    trace = manifest.get("iteration_trace", {})

    assert trace.get("journal_exists") is True
    assert trace.get("entries_count") == 1
    assert trace.get("stages_count", {}).get("review_complete") == 1
    assert trace.get("metric_improvement_round_count") == 1
    assert trace.get("metric_improvement_attempted") is True
    assert trace.get("metric_improvement_kept") == "baseline"
    assert trace.get("journal_entries_count") == 1
    assert trace.get("summary_entries_count") == 1


def test_run_manifest_includes_metric_round_records(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_id = "run_trace_rounds_01"
    state = {
        "run_id": run_id,
        "run_start_ts": "2026-02-20T00:00:00",
        "ml_improvement_round_count": 2,
        "ml_improvement_attempted": True,
        "ml_improvement_kept": "improved",
        "ml_improvement_round_history": [
            {
                "round_id": 1,
                "delta": 0.0002,
                "kept": "baseline",
                "reason": "delta_below_threshold | baseline_restored",
                "hypothesis": {"action": "APPLY", "technique": "missing_indicators", "signature": "hyp_a"},
            },
            {
                "round_id": 2,
                "delta": 0.0011,
                "kept": "improved",
                "reason": "candidate_selected",
                "hypothesis": {"action": "APPLY", "technique": "rare_grouping", "signature": "hyp_b"},
            },
        ],
    }
    run_dir = init_run_bundle(run_id, state, base_dir=str(tmp_path / "runs"), enable_tee=False)
    Path(run_dir, "contracts").mkdir(parents=True, exist_ok=True)

    manifest_path = write_run_manifest(run_id, state)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    trace = manifest.get("iteration_trace", {})
    rounds = trace.get("metric_rounds") or []

    assert trace.get("metric_rounds_count") == 2
    assert len(rounds) == 2
    assert rounds[-1].get("kept") == "improved"
    assert rounds[-1].get("hypothesis", {}).get("technique") == "rare_grouping"


def test_run_manifest_marks_metric_improvement_attempted_when_round_history_exists(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_id = "run_trace_rounds_attempted_01"
    state = {
        "run_id": run_id,
        "run_start_ts": "2026-02-20T00:00:00",
        "ml_improvement_round_count": 2,
        "ml_improvement_attempted": False,
        "ml_improvement_kept": "improved",
        "ml_improvement_round_history": [
            {"round_id": 1, "kept": "improved"},
            {"round_id": 2, "kept": "improved"},
        ],
    }

    run_dir = init_run_bundle(run_id, state, base_dir=str(tmp_path / "runs"), enable_tee=False)
    Path(run_dir, "contracts").mkdir(parents=True, exist_ok=True)
    manifest_path = write_run_manifest(run_id, state)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    trace = manifest.get("iteration_trace", {})

    assert trace.get("metric_improvement_attempted") is True
    assert trace.get("metric_rounds_count") == 2


def test_run_manifest_gates_summary_prefers_run_summary_status_over_legacy_review_verdict(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_id = "run_status_authority_01"
    csv_path = tmp_path / "input.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")
    state = {
        "run_id": run_id,
        "run_start_ts": "2026-02-20T00:00:00",
        "csv_path": str(csv_path),
        "csv_encoding": "utf-8",
        "csv_sep": ",",
        "csv_decimal": ".",
        "review_verdict_normalized": "APPROVED",
    }
    run_dir = init_run_bundle(run_id, state, base_dir=str(tmp_path / "runs"), enable_tee=False)
    report_dir = Path(run_dir) / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "run_summary.json").write_text(
        json.dumps({"status": "NEEDS_IMPROVEMENT", "failed_gates": ["runtime_failure"]}),
        encoding="utf-8",
    )
    (report_dir / "review_board_verdict.json").write_text(
        json.dumps({"status": "REJECTED", "final_review_verdict": "NEEDS_IMPROVEMENT"}),
        encoding="utf-8",
    )
    (report_dir / "output_contract_report.json").write_text(
        json.dumps({"overall_status": "ok", "missing": []}),
        encoding="utf-8",
    )

    manifest_path = write_run_manifest(run_id, state)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))

    assert manifest["status_final"] == "NEEDS_IMPROVEMENT"
    assert manifest["gates_summary"]["status"] == "NEEDS_IMPROVEMENT"


def test_run_manifest_overwrites_stale_existing_status_and_prefers_board_final_incumbent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_id = "run_manifest_authority_01"
    csv_path = tmp_path / "input.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")
    state = {
        "run_id": run_id,
        "run_start_ts": "2026-02-20T00:00:00",
        "csv_path": str(csv_path),
        "csv_encoding": "utf-8",
        "csv_sep": ",",
        "csv_decimal": ".",
        "ml_improvement_round_count": 2,
        "ml_improvement_attempted": True,
        "ml_improvement_kept": "best_attempt",
        "last_gate_context": {"feedback": "stale reviewer challenger warning"},
    }
    run_dir = init_run_bundle(run_id, state, base_dir=str(tmp_path / "runs"), enable_tee=False)
    run_dir_path = Path(run_dir)
    (run_dir_path / "run_manifest.json").write_text(
        json.dumps({"status_final": "PASS"}),
        encoding="utf-8",
    )
    artifacts_dir = run_dir_path / "artifacts" / "data"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "run_summary.json").write_text(
        json.dumps({"status": "APPROVE_WITH_WARNINGS", "failed_gates": []}),
        encoding="utf-8",
    )
    (artifacts_dir / "review_board_verdict.json").write_text(
        json.dumps(
            {
                "status": "APPROVE_WITH_WARNINGS",
                "final_incumbent_summary": "final baseline retained",
                "metric_round_finalization": {"kept": "baseline"},
            }
        ),
        encoding="utf-8",
    )
    governance_dir = run_dir_path / "report" / "governance"
    governance_dir.mkdir(parents=True, exist_ok=True)
    (governance_dir / "ml_iteration_journal.jsonl").write_text(
        "{}\n{}\n{}",
        encoding="utf-8",
    )
    (governance_dir / "ml_iteration_trace_summary.json").write_text(
        json.dumps({"entries_count": 4}),
        encoding="utf-8",
    )

    manifest_path = write_run_manifest(run_id, state)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    trace = manifest.get("iteration_trace", {})

    assert manifest["status_final"] == "APPROVE_WITH_WARNINGS"
    assert manifest["gates_summary"]["reason"] == "final baseline retained"
    assert trace["entries_count"] == 3
    assert trace["summary_entries_count"] == 4
    assert trace["metric_improvement_kept"] == "baseline"


def test_log_agent_snapshot_supports_iteration_and_attempt_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_id = "run_snapshot_trace_01"
    run_dir = init_run_bundle(run_id, {}, base_dir=str(tmp_path / "runs"), enable_tee=False)

    log_agent_snapshot(
        run_id,
        "ml_engineer",
        prompt="prompt_a",
        response="response_a",
        context={"k": "v"},
        script="print('a')",
        iteration=2,
        attempt=5,
    )

    attempt_dir = Path(run_dir) / "agents" / "ml_engineer" / "iteration_2" / "attempt_5"
    assert (attempt_dir / "prompt.txt").exists()
    assert (attempt_dir / "response.txt").exists()
    assert (attempt_dir / "context.json").exists()
    assert (attempt_dir / "script.py").exists()


def test_log_agent_snapshot_persists_subcall_history(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_id = "run_snapshot_subcalls_01"
    run_dir = init_run_bundle(run_id, {}, base_dir=str(tmp_path / "runs"), enable_tee=False)

    log_agent_snapshot(
        run_id,
        "ml_engineer",
        prompt="final prompt",
        response="final response",
        iteration=1,
        attempt=1,
        prompt_trace=[
            {
                "stage": "build_generation",
                "model_used": "model-a",
                "prompt": "build prompt",
                "response": "build response",
            },
            {
                "stage": "guardrail_repair",
                "model_used": "model-a",
                "prompt": "guard prompt",
                "response": "guard response",
            },
        ],
    )

    attempt_dir = Path(run_dir) / "agents" / "ml_engineer" / "iteration_1" / "attempt_1"
    trace_path = attempt_dir / "subcalls.json"
    assert trace_path.exists()
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert len(trace) == 2
    assert trace[0]["stage"] == "build_generation"
    assert trace[1]["stage"] == "guardrail_repair"
    assert (attempt_dir / "subcalls" / "01_build_generation_prompt.txt").exists()
    assert (attempt_dir / "subcalls" / "01_build_generation_response.txt").exists()
    assert (attempt_dir / "subcalls" / "02_guardrail_repair_prompt.txt").exists()
    assert (attempt_dir / "subcalls" / "02_guardrail_repair_response.txt").exists()
