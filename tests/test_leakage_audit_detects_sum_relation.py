import pandas as pd

from src.utils.leakage_sanity_audit import run_unsupervised_numeric_relation_audit


def test_leakage_audit_detects_sum_relation():
    data = {
        "a": list(range(50)),
        "b": list(range(50, 100)),
    }
    data["c"] = [data["a"][i] + data["b"][i] for i in range(50)]
    df = pd.DataFrame(data)

    audit = run_unsupervised_numeric_relation_audit(df, min_rows=30, tol=1e-9)
    relations = audit.get("relations", [])
    assert any(rel.get("type") == "sum" and set(rel.get("columns", [])) == {"a", "b", "c"} for rel in relations)


def test_leakage_audit_detects_categorical_identity_and_risk_flags():
    rows = 60
    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:03d}" for i in range(rows)],
            "churn_label": ["yes" if i % 2 else "no" for i in range(rows)],
            "status_alias": ["yes" if i % 2 else "no" for i in range(rows)],
        }
    )

    audit = run_unsupervised_numeric_relation_audit(df, min_rows=30, frac=0.99)
    relations = audit.get("relations", [])
    risk_flags = audit.get("risk_flags", [])

    assert any(
        rel.get("type") == "categorical_identity" and set(rel.get("columns", [])) == {"churn_label", "status_alias"}
        for rel in relations
    )
    assert any(
        flag.get("type") == "near_unique_key" and flag.get("columns") == ["customer_id"]
        for flag in risk_flags
    )
    assert any(
        flag.get("type") == "suspicious_name" and flag.get("columns") == ["churn_label"]
        for flag in risk_flags
    )
