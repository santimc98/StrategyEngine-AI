from typing import Any, Dict, List, Set


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in items:
        text = str(item or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _normalize_active_gates(active_gate_names: List[str]) -> Set[str]:
    return {
        str(item or "").strip().lower()
        for item in (active_gate_names or [])
        if str(item or "").strip()
    }


def _extract_gate_name(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("gate", "gate_name", "name", "id", "rule", "check"):
            raw = value.get(key)
            if raw:
                return str(raw).strip()
        return ""
    return str(value or "").strip()


def _fix_is_active(value: Any, active_set: Set[str]) -> bool:
    if not active_set:
        return False
    if isinstance(value, dict):
        gate_name = _extract_gate_name(value).lower()
        return bool(gate_name and gate_name in active_set)
    text = str(value or "").strip().lower()
    if not text:
        return False
    for gate in active_set:
        if gate and gate in text:
            return True
    return False


def _append_feedback_block(feedback: str, downgraded: List[str]) -> str:
    base = str(feedback or "").rstrip()
    if not downgraded:
        return base
    lines = ["NON_ACTIVE_GATE_WARNINGS:"]
    for gate in downgraded:
        lines.append(f"- {gate}")
    block = "\n".join(lines)
    if base:
        return base + "\n\n" + block
    return block


def apply_contract_first_gate_policy(
    packet: Dict[str, Any],
    active_gate_names: List[str],
    actor: str,
    hard_gate_names: Set[str] | None = None,
) -> Dict[str, Any]:
    if not isinstance(packet, dict):
        return {}

    result = dict(packet)
    active_set = _normalize_active_gates(active_gate_names)
    hard_set: Set[str] = {g.lower() for g in (hard_gate_names or set())}

    raw_failed = [_extract_gate_name(item) for item in (result.get("failed_gates") or [])]
    raw_hard = [_extract_gate_name(item) for item in (result.get("hard_failures") or [])]
    raw_hard.extend([_extract_gate_name(item) for item in (result.get("hard_blockers") or [])])

    failed_active: List[str] = []
    hard_active: List[str] = []
    soft_active: List[str] = []
    downgraded_gates: List[str] = []

    for gate in raw_failed:
        if not gate:
            continue
        if gate.lower() in active_set:
            failed_active.append(gate)
            if gate.lower() in hard_set:
                hard_active.append(gate)
            else:
                soft_active.append(gate)
        else:
            downgraded_gates.append(gate)

    for gate in raw_hard:
        if not gate:
            continue
        if gate.lower() in active_set:
            if gate not in hard_active:
                hard_active.append(gate)
            if gate not in failed_active:
                failed_active.append(gate)
        else:
            downgraded_gates.append(gate)

    filtered_fixes: List[Any] = []
    downgraded_fixes: List[str] = []
    for fix in (result.get("required_fixes") or []):
        if _fix_is_active(fix, active_set):
            filtered_fixes.append(fix)
        else:
            fix_name = _extract_gate_name(fix) if isinstance(fix, dict) else str(fix or "").strip()
            if fix_name:
                downgraded_fixes.append(fix_name)

    downgraded_gates = _dedupe_keep_order(downgraded_gates + downgraded_fixes)
    failed_active = _dedupe_keep_order(failed_active)
    hard_active = _dedupe_keep_order(hard_active)
    soft_active = _dedupe_keep_order(soft_active)

    result["failed_gates"] = failed_active
    result["hard_failures"] = hard_active
    if "hard_blockers" in result:
        result["hard_blockers"] = list(hard_active)
    result["required_fixes"] = filtered_fixes

    warnings = [str(item) for item in (result.get("warnings") or []) if str(item).strip()]
    if soft_active:
        soft_note = f"{actor}: soft gate failures kept as advisory -> {sorted([g.lower() for g in soft_active])}"
        if soft_note not in warnings:
            warnings.append(soft_note)
    if downgraded_gates:
        downgrade_note = f"{actor}: downgraded non-active gates -> {sorted([g.lower() for g in downgraded_gates])}"
        if downgrade_note not in warnings:
            warnings.append(downgrade_note)
        result["feedback"] = _append_feedback_block(result.get("feedback"), downgraded_gates)
    result["warnings"] = warnings

    status = str(result.get("status") or "").strip().upper()
    approved_statuses = {"APPROVED", "APPROVE_WITH_WARNINGS", "APPROVED_WITH_WARNINGS", "PASS", "OK", "SUCCESS"}
    rejected_statuses = {"REJECTED", "NEEDS_IMPROVEMENT", "FAIL", "FAILED", "ERROR", "CRASH"}

    has_hard_blockers = bool(hard_active)
    has_only_soft_failures = bool(failed_active) and not hard_active
    if status in approved_statuses and has_hard_blockers:
        result["status"] = "REJECTED"
    elif status in approved_statuses and has_only_soft_failures:
        result["status"] = "APPROVE_WITH_WARNINGS"
    elif status in rejected_statuses and not has_hard_blockers and downgraded_gates:
        result["status"] = "APPROVE_WITH_WARNINGS"
    elif status in rejected_statuses and not has_hard_blockers and has_only_soft_failures:
        result["status"] = "APPROVE_WITH_WARNINGS"

    return result
