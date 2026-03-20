from src.utils.sandbox_deps import (
    check_dependency_precheck,
    classify_dependency_support,
    cloudrun_imports_from_code,
)


def test_dependency_precheck_blocks_banned_import():
    code = "import pulp\n"
    result = check_dependency_precheck(code, required_dependencies=[])
    assert "pulp" in result["banned"]
    assert "pulp" in result.get("suggestions", {})


def test_dependency_precheck_allows_base_imports():
    code = "import pandas\nfrom sklearn.linear_model import LogisticRegression\nimport json\nimport statsmodels\n"
    result = check_dependency_precheck(code, required_dependencies=[])
    assert result["blocked"] == []
    assert result["banned"] == []


def test_dependency_precheck_allows_extended_when_contract_requests():
    code = "import rapidfuzz\n"
    result = check_dependency_precheck(code, required_dependencies=["rapidfuzz"])
    assert result["blocked"] == []
    assert result["banned"] == []


def test_dependency_precheck_blocks_extended_when_not_requested():
    code = "import rapidfuzz\n"
    result = check_dependency_precheck(code, required_dependencies=[])
    assert "rapidfuzz" in result["blocked"]


def test_dependency_precheck_blocks_torch_on_local():
    code = "import torch\n"
    result = check_dependency_precheck(code, required_dependencies=[], backend_profile="local")
    assert "torch" in result["banned"]


def test_dependency_precheck_allows_torch_on_cloudrun():
    code = "import torch\n"
    result = check_dependency_precheck(code, required_dependencies=[], backend_profile="cloudrun")
    assert "torch" not in result["banned"]
    assert "torch" not in result["blocked"]


def test_dependency_precheck_allows_lightgbm_on_cloudrun_without_contract_hint():
    code = "import lightgbm\n"
    result = check_dependency_precheck(code, required_dependencies=[], backend_profile="cloudrun")
    assert "lightgbm" not in result["banned"]
    assert "lightgbm" not in result["blocked"]


def test_dependency_precheck_allows_catboost_on_cloudrun_without_contract_hint():
    code = "import catboost\n"
    result = check_dependency_precheck(code, required_dependencies=[], backend_profile="cloudrun")
    assert "catboost" not in result["banned"]
    assert "catboost" not in result["blocked"]


def test_dependency_precheck_allows_joblib_on_cloudrun():
    code = "import joblib\n"
    result = check_dependency_precheck(code, required_dependencies=[], backend_profile="cloudrun")
    assert "joblib" not in result["banned"]
    assert "joblib" not in result["blocked"]


def test_cloudrun_import_detection_handles_markdown_fences():
    code = "```python\nfrom transformers import AutoModel\nimport torch\n```"
    imports = cloudrun_imports_from_code(code)
    assert "transformers" in imports
    assert "torch" in imports


def test_dependency_precheck_allows_survival_libs_on_cloudrun():
    code = "import lifelines\nfrom sksurv.linear_model import CoxPHSurvivalAnalysis\n"
    result = check_dependency_precheck(code, required_dependencies=[], backend_profile="cloudrun")
    assert result["blocked"] == []
    assert result["banned"] == []


def test_classify_dependency_support_normalizes_survival_aliases():
    support = classify_dependency_support(
        ["lifelines", "scikit-survival"],
        backend_profile="cloudrun",
    )
    assert support["blocked"] == []
    assert support["banned"] == []
    assert "lifelines" in support["supported"]
    assert "sksurv" in support["supported"]
