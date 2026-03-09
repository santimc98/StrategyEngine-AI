from src.graph.graph import _apply_review_consistency_guard, _build_code_review_diagnostics


def test_build_code_review_diagnostics_marks_runtime_and_missing_outputs() -> None:
    diagnostics = _build_code_review_diagnostics(
        runtime_failure_detected=True,
        runtime_terminal=False,
        execution_output="Traceback ...",
        output_contract_report={"overall_status": "error", "missing": ["outputs/a.csv"]},
        artifact_index=[],
    )

    blockers = diagnostics.get("hard_blockers") or []
    assert "runtime_failure" in blockers
    assert "output_contract_error" in blockers
    assert "contract_required_artifacts_missing" in blockers


def test_apply_review_consistency_guard_preserves_approved_with_blocker_signal() -> None:
    result = {
        "status": "APPROVED",
        "feedback": "all good",
        "failed_gates": [],
        "required_fixes": [],
    }
    diagnostics = {
        "hard_blockers": ["runtime_failure", "contract_required_artifacts_missing"],
    }

    guarded = _apply_review_consistency_guard(result, diagnostics, actor="reviewer")

    assert guarded["status"] == "APPROVED"
    signals = guarded.get("consistency_signals") or {}
    signal = signals.get("deterministic_blockers") or {}
    assert "runtime_failure" in (signal.get("hard_blockers") or [])
    assert "contract_required_artifacts_missing" in (signal.get("hard_blockers") or [])
    assert signal.get("preserve_llm_status") is True
    assert any("CONTEXT_GUARD" in item for item in (guarded.get("warnings") or []))


def test_apply_review_consistency_guard_keeps_non_approved_status() -> None:
    result = {
        "status": "REJECTED",
        "feedback": "already rejected",
        "failed_gates": ["qa_gate"],
        "required_fixes": ["fix qa gate"],
    }
    diagnostics = {"hard_blockers": ["runtime_failure"]}

    guarded = _apply_review_consistency_guard(result, diagnostics, actor="qa_reviewer")

    assert guarded["status"] == "REJECTED"
    assert guarded["failed_gates"] == ["qa_gate"]
    signal = (guarded.get("consistency_signals") or {}).get("deterministic_blockers") or {}
    assert "runtime_failure" in (signal.get("hard_blockers") or [])
