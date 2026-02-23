from src.agents.qa_reviewer import collect_static_qa_facts, run_static_qa_checks


EVALUATION_SPEC = {
    "requires_target": True,
    "qa_gates": [{"name": "target_variance_guard", "severity": "HARD", "params": {}}],
}


def _assert_guard_detected_and_not_rejected(code: str) -> None:
    facts = collect_static_qa_facts(code)
    assert facts["has_variance_guard"] is True

    result = run_static_qa_checks(code, evaluation_spec=EVALUATION_SPEC)
    assert result is not None
    assert "target_variance_guard" not in (result.get("failed_gates") or [])
    assert result.get("status") in {"PASS", "WARN"}


def test_variance_guard_detects_nunique_alias_le_one() -> None:
    code = """
import pandas as pd
df = pd.read_csv("data/cleaned_data.csv")
valid_targets = df["target"].dropna()
n_unique = valid_targets.nunique()
if n_unique <= 1:
    raise ValueError("Target has no variance; cannot train.")
"""
    _assert_guard_detected_and_not_rejected(code)


def test_variance_guard_detects_nunique_alias_lt_two() -> None:
    code = """
import pandas as pd
df = pd.read_csv("data/cleaned_data.csv")
target_count = df["target"].nunique()
if target_count < 2:
    raise ValueError("Target has no variance; cannot train.")
"""
    _assert_guard_detected_and_not_rejected(code)


def test_variance_guard_detects_direct_nunique_le_one_regression() -> None:
    code = """
import pandas as pd
df = pd.read_csv("data/cleaned_data.csv")
if df["target"].nunique() <= 1:
    raise ValueError("Target has no variance; cannot train.")
"""
    _assert_guard_detected_and_not_rejected(code)


def test_variance_guard_detects_var_alias_eq_zero() -> None:
    code = """
import pandas as pd
df = pd.read_csv("data/cleaned_data.csv")
target_var = df["target"].var()
if target_var == 0:
    raise ValueError("Target variance is zero; cannot train.")
"""
    _assert_guard_detected_and_not_rejected(code)


def test_variance_guard_detects_unique_len_alias_le_one() -> None:
    code = """
import pandas as pd
df = pd.read_csv("data/cleaned_data.csv")
target_num = pd.to_numeric(df["target"], errors="coerce")
unique_targets = target_num.dropna().unique()
if len(unique_targets) <= 1:
    raise ValueError("Target has no variance; cannot train.")
"""
    _assert_guard_detected_and_not_rejected(code)


def test_variance_guard_detects_assert_nunique_alias_gt_one() -> None:
    code = """
import pandas as pd
df = pd.read_csv("data/cleaned_data.csv")
n_unique = df["target"].nunique()
assert n_unique > 1, "Target has no variance; cannot train."
"""
    _assert_guard_detected_and_not_rejected(code)


def test_variance_guard_detects_deferred_raise_via_failure_container() -> None:
    code = """
import pandas as pd
df = pd.read_csv("data/cleaned_data.csv")
failures = []
target_nunique = df["target"].nunique()
if target_nunique <= 1:
    failures.append("target_variance_guard")

if failures:
    raise ValueError("Validation failed before training.")
"""
    _assert_guard_detected_and_not_rejected(code)
