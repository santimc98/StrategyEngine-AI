from typing import Any, Dict, List, Optional


import re


def _as_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item]


def _extract_primary_target(semantics: Dict[str, Any]) -> str | None:
    primary = semantics.get("primary_target")
    if isinstance(primary, str) and primary.strip():
        return primary
    target_info = semantics.get("target_analysis")
    if isinstance(target_info, dict):
        primary = target_info.get("primary_target")
        if isinstance(primary, str) and primary.strip():
            return primary
    return None


def _extract_target_status(semantics: Dict[str, Any]) -> str | None:
    status = semantics.get("target_status")
    if isinstance(status, str) and status.strip():
        return status.strip().lower()
    target_info = semantics.get("target_analysis")
    if isinstance(target_info, dict):
        status = target_info.get("target_status")
        if isinstance(status, str) and status.strip():
            return status.strip().lower()
    return None


def _extract_recommended_primary_target(semantics: Dict[str, Any]) -> str | None:
    recommended = semantics.get("recommended_primary_target")
    if isinstance(recommended, str) and recommended.strip():
        return recommended.strip()
    target_info = semantics.get("target_analysis")
    if isinstance(target_info, dict):
        recommended = target_info.get("recommended_primary_target")
        if isinstance(recommended, str) and recommended.strip():
            return recommended.strip()
    return None


def _extract_target_status_reason(semantics: Dict[str, Any]) -> str | None:
    reason = semantics.get("target_status_reason")
    if isinstance(reason, str) and reason.strip():
        return reason.strip()
    target_info = semantics.get("target_analysis")
    if isinstance(target_info, dict):
        reason = target_info.get("target_status_reason")
        if isinstance(reason, str) and reason.strip():
            return reason.strip()
    return None


def _extract_target_columns(semantics: Dict[str, Any]) -> List[str]:
    targets: List[str] = []

    def _extend(values: Any) -> None:
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list):
            return
        for value in values:
            token = str(value or "").strip()
            if token and token not in targets:
                targets.append(token)

    _extend(semantics.get("target_columns"))
    _extend(semantics.get("primary_targets"))
    target_info = semantics.get("target_analysis")
    if isinstance(target_info, dict):
        _extend(target_info.get("target_columns"))
        _extend(target_info.get("primary_targets"))

    primary_target = _extract_primary_target(semantics)
    if primary_target and primary_target not in targets:
        targets.insert(0, primary_target)
    return targets


def _normalize_target_token(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    token = value.strip().strip("`").strip("'").strip('"').strip()
    if not token:
        return None
    if token.lower() in {"missing", "none", "null", "n/a", "na"}:
        return None
    return token


def _extract_preliminary_steward_primary_target(steward_summary: Dict[str, Any]) -> str | None:
    if not isinstance(steward_summary, dict):
        return None

    for key in ("recommended_primary_target", "primary_target"):
        token = _normalize_target_token(steward_summary.get(key))
        if token:
            return token

    summary_text = str(steward_summary.get("summary") or "").strip()
    if not summary_text:
        return None

    patterns = [
        r"recommended\s+primary\s+target\s*:\s*`?([A-Za-z0-9_]+)`?",
        r"primary\s+target\s*:\s*`?([A-Za-z0-9_]+)`?",
        r"recommended\s+target\s*:\s*`?([A-Za-z0-9_]+)`?",
    ]
    for pattern in patterns:
        match = re.search(pattern, summary_text, flags=re.IGNORECASE)
        if not match:
            continue
        token = _normalize_target_token(match.group(1))
        if token:
            return token
    return None


def _extract_contract_primary_target(contract: Dict[str, Any]) -> str | None:
    if not isinstance(contract, dict):
        return None

    def _from_section(section: Any) -> str | None:
        if not isinstance(section, dict):
            return None
        for key in ("primary_target", "target_column", "label_column"):
            token = _normalize_target_token(section.get(key))
            if token:
                return token
        for key in ("label_columns", "target_columns"):
            values = section.get(key)
            if isinstance(values, str):
                values = [values]
            if isinstance(values, list):
                for item in values:
                    token = _normalize_target_token(item)
                    if token:
                        return token
        return None

    for section_name in ("task_semantics", "evaluation_spec", "validation_requirements"):
        token = _from_section(contract.get(section_name))
        if token:
            return token

    return _from_section(contract)


def build_target_lineage_summary(
    steward_summary: Dict[str, Any] | None,
    dataset_semantics: Dict[str, Any] | None,
    contract: Dict[str, Any] | None,
) -> Dict[str, Any]:
    steward_summary = steward_summary if isinstance(steward_summary, dict) else {}
    dataset_semantics = dataset_semantics if isinstance(dataset_semantics, dict) else {}
    contract = contract if isinstance(contract, dict) else {}

    preliminary_target = _extract_preliminary_steward_primary_target(steward_summary)
    validated_target = _extract_primary_target(dataset_semantics)
    target_status = _extract_target_status(dataset_semantics)
    recommended_target = _extract_recommended_primary_target(dataset_semantics)
    target_status_reason = _extract_target_status_reason(dataset_semantics)
    final_contract_target = _extract_contract_primary_target(contract)
    preliminary_summary_excerpt = str(steward_summary.get("summary") or "").strip()[:1200]

    preliminary_conflict = bool(
        preliminary_target and validated_target and preliminary_target != validated_target
    )
    contract_differs_from_validated = bool(
        final_contract_target and validated_target and final_contract_target != validated_target
    )
    contract_matches_preliminary = bool(
        final_contract_target and preliminary_target and final_contract_target == preliminary_target
    )
    contract_matches_recommendation = bool(
        final_contract_target and recommended_target and final_contract_target == recommended_target
    )

    lineage_status = "aligned"
    if preliminary_conflict or contract_differs_from_validated:
        lineage_status = "diverged"
    elif target_status in {"questioned", "invalid"}:
        lineage_status = "flagged"

    summary_parts: List[str] = []
    if preliminary_target:
        summary_parts.append(f"Preliminary steward assessment named {preliminary_target}.")
    if validated_target:
        validated_line = f"Validated steward semantics kept {validated_target}"
        if target_status:
            validated_line += f" with status {target_status}"
        validated_line += "."
        summary_parts.append(validated_line)
    if recommended_target and recommended_target != validated_target:
        summary_parts.append(f"Validated steward recommendation pointed to {recommended_target}.")
    if final_contract_target:
        summary_parts.append(f"Final execution contract selected {final_contract_target}.")
    if preliminary_conflict:
        summary_parts.append(
            "The preliminary steward narrative and the validated steward semantics disagreed on the target."
        )
    if contract_differs_from_validated:
        summary_parts.append(
            "The final contract target diverged from the validated steward semantics and should be narrated as an explicit downstream decision."
        )
    if target_status_reason:
        summary_parts.append(f"Validated target rationale: {target_status_reason}")

    return {
        "lineage_status": lineage_status,
        "preliminary_steward_target": preliminary_target,
        "validated_steward_target": validated_target,
        "validated_steward_target_status": target_status,
        "validated_steward_recommended_target": recommended_target,
        "validated_steward_target_reason": target_status_reason,
        "final_contract_target": final_contract_target,
        "preliminary_summary_conflicts_with_validated_semantics": preliminary_conflict,
        "contract_differs_from_validated_steward": contract_differs_from_validated,
        "contract_matches_preliminary_steward": contract_matches_preliminary,
        "contract_matches_steward_recommendation": contract_matches_recommendation,
        "preliminary_summary_excerpt": preliminary_summary_excerpt,
        "lineage_summary": " ".join(summary_parts).strip(),
    }


def summarize_dataset_semantics(
    semantics: Dict[str, Any],
    training_mask: Optional[Dict[str, Any]] = None,
    max_lines: int = 18,
) -> str:
    semantics = semantics if isinstance(semantics, dict) else {}
    training_mask = training_mask if isinstance(training_mask, dict) else {}

    primary_target = _extract_primary_target(semantics)
    target_status = _extract_target_status(semantics)
    recommended_primary_target = _extract_recommended_primary_target(semantics)
    target_status_reason = _extract_target_status_reason(semantics)
    target_columns = _extract_target_columns(semantics)
    target_info = semantics.get("target_analysis") if isinstance(semantics.get("target_analysis"), dict) else {}
    partial_labels = target_info.get("partial_label_detected") if isinstance(target_info, dict) else None
    null_frac_exact = target_info.get("target_null_frac_exact") if isinstance(target_info, dict) else None
    missing_exact = target_info.get("target_missing_count_exact") if isinstance(target_info, dict) else None
    total_exact = target_info.get("target_total_count_exact") if isinstance(target_info, dict) else None

    split_candidates = _as_list(semantics.get("split_candidates"))
    partition_info = semantics.get("partition_analysis") if isinstance(semantics.get("partition_analysis"), dict) else {}
    if not split_candidates:
        split_candidates = _as_list(partition_info.get("partition_columns"))
    id_candidates = _as_list(semantics.get("id_candidates"))

    lines: List[str] = []
    lines.append("DATASET_SEMANTICS_SUMMARY:")
    lines.append(f"- primary_target: {primary_target or 'missing'}")
    if target_status:
        lines.append(f"- target_status: {target_status}")
    if recommended_primary_target:
        lines.append(f"- recommended_primary_target: {recommended_primary_target}")
    if target_status_reason:
        lines.append(f"- target_status_reason: {target_status_reason}")
    if target_columns:
        lines.append(f"- target_columns: {target_columns[:8]}")
    if partial_labels is not None:
        lines.append(f"- partial_label_detected: {bool(partial_labels)}")
    if null_frac_exact is not None:
        ratio = round(float(null_frac_exact), 6)
        if total_exact is not None and missing_exact is not None:
            lines.append(f"- target_null_frac_exact: {ratio} ({missing_exact}/{total_exact})")
        else:
            lines.append(f"- target_null_frac_exact: {ratio}")
    if split_candidates:
        lines.append(f"- split_candidates: {split_candidates[:5]}")
    if id_candidates:
        lines.append(f"- id_candidates: {id_candidates[:5]}")
    if training_mask:
        lines.append(f"- training_rows_rule: {training_mask.get('training_rows_rule', '')}")
        lines.append(f"- scoring_rows_rule_primary: {training_mask.get('scoring_rows_rule_primary', '')}")
        secondary = training_mask.get("scoring_rows_rule_secondary")
        if secondary:
            lines.append(f"- scoring_rows_rule_secondary: {secondary}")
        rationale = training_mask.get("rationale")
        if isinstance(rationale, list) and rationale:
            lines.append(f"- rationale: {rationale[:3]}")

    return "\n".join(lines[:max_lines])
