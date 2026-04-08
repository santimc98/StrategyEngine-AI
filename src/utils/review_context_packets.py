import re
from typing import Any, Dict, List


_TOKEN_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "into",
    "over",
    "under",
    "only",
    "must",
    "should",
    "would",
    "could",
    "when",
    "then",
    "than",
    "have",
    "has",
    "had",
    "using",
    "used",
    "rows",
    "row",
    "gate",
    "hard",
    "soft",
    "code",
    "json",
    "path",
    "hint",
    "review",
    "reviewer",
    "qa",
    "contract",
    "active",
    "required",
    "candidate",
    "restored",
    "known",
    "risk",
    "risks",
    "evidence",
    "summary",
    "context",
    "status",
    "failed",
    "fix",
    "fixes",
    "change",
    "changes",
    "attempt",
}


def _truncate_text(value: Any, max_len: int = 220) -> str:
    text = str(value or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _normalize_gate_spec(item: Any) -> Dict[str, Any]:
    if not isinstance(item, dict):
        if isinstance(item, str) and item.strip():
            return {"name": item.strip(), "severity": "HARD", "params": {}}
        return {}
    name = (
        item.get("name")
        or item.get("id")
        or item.get("gate")
        or item.get("metric")
        or item.get("check")
        or item.get("rule")
        or item.get("title")
        or item.get("label")
    )
    if not name:
        return {}
    severity = str(item.get("severity") or "HARD").strip().upper() or "HARD"
    if severity not in {"HARD", "SOFT"}:
        severity = "HARD"
    params = item.get("params")
    if not isinstance(params, dict):
        params = {}
    return {"name": str(name).strip(), "severity": severity, "params": params}


def _sanitize_scalar(value: Any) -> Any:
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, str):
        return _truncate_text(value, max_len=120)
    return None


def _sanitize_params(params: Dict[str, Any], max_items: int = 6) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not isinstance(params, dict):
        return out
    for key, value in params.items():
        if len(out) >= max_items:
            break
        normalized_key = str(key or "").strip()
        if not normalized_key:
            continue
        if isinstance(value, list):
            items: List[Any] = []
            for item in value[:4]:
                scalar = _sanitize_scalar(item)
                if scalar is not None:
                    items.append(scalar)
            if items:
                out[normalized_key] = items
            continue
        scalar = _sanitize_scalar(value)
        if scalar is not None:
            out[normalized_key] = scalar
    return out


def _append_unique_entry(entries: List[Dict[str, Any]], item: Dict[str, Any], max_items: int) -> None:
    if not isinstance(item, dict):
        return
    normalized = {
        "gate": str(item.get("gate") or "").strip(),
        "concern": _truncate_text(item.get("concern") or "", max_len=220),
        "source": str(item.get("source") or "missing").strip() or "missing",
        "restored_candidate_risk": bool(item.get("restored_candidate_risk")),
    }
    if not normalized["concern"]:
        return
    if normalized in entries:
        return
    if len(entries) >= max_items:
        return
    entries.append(normalized)


def _extract_keywords(value: Any) -> set[str]:
    tokens: set[str] = set()
    if value is None:
        return tokens
    if isinstance(value, (int, float)):
        tokens.add(str(value))
        return tokens
    text = str(value or "").strip()
    if not text:
        return tokens
    for raw in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}|\d+(?:\.\d+)?", text):
        token = raw.strip()
        lower = token.lower()
        if lower in _TOKEN_STOPWORDS:
            continue
        if len(lower) < 4 and not any(ch.isdigit() for ch in lower):
            continue
        tokens.add(token)
    return tokens


def _extract_known_candidate_risks(
    context_blocks: List[Any],
    hard_gate_names: set[str],
    max_items: int = 8,
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []

    def _gate_or_blank(name: Any) -> str:
        text = str(name or "").strip()
        return text if text.lower() in hard_gate_names else ""

    for block in context_blocks:
        if not isinstance(block, dict):
            continue

        review_history = block.get("review_history_context")
        if isinstance(review_history, dict):
            restored_recently = bool(review_history.get("best_attempt_restored_recently"))
            if restored_recently:
                _append_unique_entry(
                    entries,
                    {
                        "concern": "Candidate was recently restored from a previous best attempt; re-check previously surfaced blockers before approval.",
                        "source": "review_history_context#best_attempt_restored_recently",
                        "restored_candidate_risk": True,
                    },
                    max_items,
                )
            history_tail = review_history.get("feedback_history_tail")
            if isinstance(history_tail, list):
                for idx, item in enumerate(history_tail[-4:]):
                    text = _truncate_text(item, max_len=220)
                    if not text:
                        continue
                    _append_unique_entry(
                        entries,
                        {
                            "concern": text,
                            "source": f"review_history_context#feedback_history_tail[{idx}]",
                            "restored_candidate_risk": "best_attempt_restored" in text.lower(),
                        },
                        max_items,
                    )
            last_gate_context = review_history.get("last_gate_context")
            if isinstance(last_gate_context, dict):
                failed_gates = last_gate_context.get("failed_gates")
                if isinstance(failed_gates, list):
                    for gate in failed_gates[:6]:
                        gate_name = _gate_or_blank(gate)
                        _append_unique_entry(
                            entries,
                            {
                                "gate": gate_name,
                                "concern": f"Previous review failed gate '{gate_name or str(gate)}'; verify the current candidate truly resolves it.",
                                "source": "review_history_context#last_gate_context.failed_gates",
                                "restored_candidate_risk": restored_recently,
                            },
                            max_items,
                        )
                required_fixes = last_gate_context.get("required_fixes")
                if isinstance(required_fixes, list):
                    for idx, fix in enumerate(required_fixes[:6]):
                        _append_unique_entry(
                            entries,
                            {
                                "concern": str(fix or "").strip(),
                                "source": f"review_history_context#last_gate_context.required_fixes[{idx}]",
                                "restored_candidate_risk": restored_recently,
                            },
                            max_items,
                        )

        last_gate_context = block.get("last_gate_context")
        if isinstance(last_gate_context, dict):
            failed_gates = last_gate_context.get("failed_gates")
            if isinstance(failed_gates, list):
                for gate in failed_gates[:6]:
                    gate_name = _gate_or_blank(gate)
                    _append_unique_entry(
                        entries,
                        {
                            "gate": gate_name,
                            "concern": f"Latest gate context marked '{gate_name or str(gate)}' as unresolved.",
                            "source": "last_gate_context#failed_gates",
                        },
                        max_items,
                    )
            required_fixes = last_gate_context.get("required_fixes")
            if isinstance(required_fixes, list):
                for idx, fix in enumerate(required_fixes[:6]):
                    _append_unique_entry(
                        entries,
                        {
                            "concern": str(fix or "").strip(),
                            "source": f"last_gate_context#required_fixes[{idx}]",
                        },
                        max_items,
                    )

        iteration_handoff = block.get("iteration_handoff")
        if isinstance(iteration_handoff, dict):
            failed_gates = iteration_handoff.get("failed_gates")
            if isinstance(failed_gates, list):
                for gate in failed_gates[:6]:
                    gate_name = _gate_or_blank(gate)
                    _append_unique_entry(
                        entries,
                        {
                            "gate": gate_name,
                            "concern": f"Iteration handoff still tracks '{gate_name or str(gate)}' as a blocker for this candidate.",
                            "source": "iteration_handoff#failed_gates",
                        },
                        max_items,
                    )
            for key in ("patch_objectives", "required_fixes"):
                values = iteration_handoff.get(key)
                if isinstance(values, list):
                    for idx, item in enumerate(values[:6]):
                        _append_unique_entry(
                            entries,
                            {
                                "concern": str(item or "").strip(),
                                "source": f"iteration_handoff#{key}[{idx}]",
                            },
                            max_items,
                        )

        diagnostics = block.get("execution_diagnostics")
        if isinstance(diagnostics, dict):
            blockers = diagnostics.get("hard_blockers")
            if isinstance(blockers, list):
                for blocker in blockers[:6]:
                    _append_unique_entry(
                        entries,
                        {
                            "gate": _gate_or_blank(blocker),
                            "concern": f"Execution diagnostics still report blocker '{blocker}'.",
                            "source": "execution_diagnostics#hard_blockers",
                        },
                        max_items,
                    )

    return entries


def _build_code_lines_of_interest(
    code: str,
    code_path_hint: str,
    active_hard_gates_summary: List[Dict[str, Any]],
    known_candidate_risks: List[Dict[str, Any]],
    max_items: int = 8,
) -> List[Dict[str, Any]]:
    lines = str(code or "").splitlines()
    if not lines:
        return []

    keywords: set[str] = set()
    for gate in active_hard_gates_summary:
        keywords.update(_extract_keywords(gate.get("name")))
        params = gate.get("params")
        if isinstance(params, dict):
            for key, value in params.items():
                keywords.update(_extract_keywords(key))
                if isinstance(value, list):
                    for item in value:
                        keywords.update(_extract_keywords(item))
                else:
                    keywords.update(_extract_keywords(value))
    for risk in known_candidate_risks:
        keywords.update(_extract_keywords(risk.get("concern")))
        keywords.update(_extract_keywords(risk.get("gate")))

    ranked: List[tuple[int, int, str, List[str]]] = []
    lowered_keywords = {token.lower(): token for token in keywords}
    for line_number, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        lower_line = stripped.lower()
        matched: List[str] = []
        score = 0
        for lower_token, original in lowered_keywords.items():
            if lower_token in lower_line:
                matched.append(original)
                if any(ch.isdigit() for ch in lower_token) or "_" in lower_token:
                    score += 4
                elif len(lower_token) >= 8:
                    score += 3
                else:
                    score += 2
        if score <= 0:
            continue
        ranked.append((score, line_number, stripped, sorted(set(matched), key=str.lower)))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    out: List[Dict[str, Any]] = []
    seen_lines: set[int] = set()
    normalized_path = str(code_path_hint or "script.py").strip() or "script.py"
    for score, line_number, snippet, matched_terms in ranked:
        if line_number in seen_lines:
            continue
        seen_lines.add(line_number)
        out.append(
            {
                "path": normalized_path,
                "line": line_number,
                "snippet": _truncate_text(snippet, max_len=200),
                "matched_terms": matched_terms[:6],
                "score": score,
            }
        )
        if len(out) >= max_items:
            break
    return out


def build_review_context_packet(
    code: str,
    gate_specs: List[Any] | None,
    *,
    code_path_hint: str = "",
    context_blocks: List[Any] | None = None,
    max_items: int = 8,
) -> Dict[str, Any]:
    normalized_gates: List[Dict[str, Any]] = []
    seen_gates: set[str] = set()
    for item in gate_specs or []:
        spec = _normalize_gate_spec(item)
        if not spec:
            continue
        key = str(spec.get("name") or "").strip().lower()
        if not key or key in seen_gates:
            continue
        seen_gates.add(key)
        normalized_gates.append(spec)

    active_hard_gates_summary: List[Dict[str, Any]] = []
    for spec in normalized_gates:
        if str(spec.get("severity") or "HARD").upper() != "HARD":
            continue
        active_hard_gates_summary.append(
            {
                "name": spec.get("name"),
                "params": _sanitize_params(spec.get("params") if isinstance(spec.get("params"), dict) else {}),
            }
        )
        if len(active_hard_gates_summary) >= max_items:
            break

    hard_gate_names = {
        str(item.get("name") or "").strip().lower()
        for item in active_hard_gates_summary
        if isinstance(item, dict) and item.get("name")
    }
    known_candidate_risks = _extract_known_candidate_risks(
        context_blocks or [],
        hard_gate_names,
        max_items=max_items,
    )
    known_restored_candidate_risks = [
        item for item in known_candidate_risks if bool(item.get("restored_candidate_risk"))
    ][:max_items]
    code_lines_of_interest = _build_code_lines_of_interest(
        code,
        code_path_hint=code_path_hint,
        active_hard_gates_summary=active_hard_gates_summary,
        known_candidate_risks=known_candidate_risks,
        max_items=max_items,
    )

    hard_blocker_packet = {
        "active_hard_gates_summary": active_hard_gates_summary,
        "known_candidate_risks": known_candidate_risks,
        "known_restored_candidate_risks": known_restored_candidate_risks,
        "code_lines_of_interest": code_lines_of_interest,
    }
    return hard_blocker_packet
