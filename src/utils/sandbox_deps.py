import ast
import sys
import re
from typing import Dict, Iterable, List, Set

BASE_ALLOWLIST = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "joblib",
    "statsmodels",
    "matplotlib",
    "seaborn",
    "pyarrow",
    "openpyxl",
    "duckdb",
    "sqlalchemy",
    "dateutil",
    "pytz",
    "tqdm",
    "yaml",
]
EXTENDED_ALLOWLIST = ["rapidfuzz", "plotly", "pydantic", "pandera", "networkx"]
CLOUDRUN_NATIVE_ALLOWLIST = [
    "torch",
    "transformers",
    "tokenizers",
    "datasets",
    "accelerate",
    "sentence_transformers",
]
CLOUDRUN_OPTIONAL_ALLOWLIST = [
    "xgboost",
    "lightgbm",
    "catboost",
    "shap",
    "optuna",
    "imblearn",
    "category_encoders",
    "plotly",
    "rapidfuzz",
    "pydantic",
    "pandera",
    "networkx",
    "sentencepiece",
    "huggingface_hub",
]
# Unsupported regardless of backend (either incompatible or out-of-scope for this product runtime)
BANNED_ALWAYS_ALLOWLIST = ["tensorflow", "keras", "pyspark", "spacy", "prophet", "cvxpy", "pulp", "fuzzywuzzy"]
# Imports explicitly blocked on E2B due to runtime constraints (8GB RAM / lightweight profile)
E2B_HEAVY_BLOCKLIST = ["torch", "transformers", "tokenizers", "datasets", "accelerate", "sentence_transformers"]

PIP_BASE = [
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "joblib",
    "statsmodels",
    "matplotlib",
    "seaborn",
    "pyarrow",
    "openpyxl",
    "duckdb",
    "sqlalchemy",
    "python-dateutil",
    "pytz",
    "tqdm",
    "pyyaml",
]
PIP_EXTENDED = {
    "rapidfuzz": "rapidfuzz",
    "plotly": "plotly",
    "pydantic": "pydantic",
    "pandera": "pandera",
    "networkx": "networkx",
}
PIP_CLOUDRUN_NATIVE = {
    "torch": "torch",
    "transformers": "transformers",
    "tokenizers": "tokenizers",
    "datasets": "datasets",
    "accelerate": "accelerate",
    "sentence_transformers": "sentence-transformers",
}
PIP_CLOUDRUN_OPTIONAL = {
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "catboost": "catboost",
    "shap": "shap",
    "optuna": "optuna",
    "imblearn": "imbalanced-learn",
    "category_encoders": "category_encoders",
    "plotly": "plotly",
    "rapidfuzz": "rapidfuzz",
    "pydantic": "pydantic",
    "pandera": "pandera",
    "networkx": "networkx",
    "sentencepiece": "sentencepiece",
    "huggingface_hub": "huggingface-hub",
}

_BACKEND_ALIASES = {
    "e2b": "e2b",
    "default": "e2b",
    "local": "e2b",
    "cloudrun": "cloudrun",
    "cloud_run": "cloudrun",
    "heavy": "cloudrun",
    "heavy_runner": "cloudrun",
}


def _stdlib_modules() -> Set[str]:
    if hasattr(sys, "stdlib_module_names"):
        return set(sys.stdlib_module_names)
    return {
        "abc", "argparse", "asyncio", "base64", "collections", "contextlib", "csv",
        "dataclasses", "datetime", "enum", "functools", "glob", "hashlib", "itertools",
        "json", "logging", "math", "os", "pathlib", "random", "re", "statistics",
        "string", "sys", "time", "typing", "uuid", "warnings",
    }


def extract_import_roots(code: str) -> Set[str]:
    text = str(code or "")
    try:
        tree = ast.parse(text)
    except Exception:
        # Retry with markdown code fences stripped to recover imports from LLM-formatted snippets.
        text = re.sub(r"^\s*```[a-zA-Z0-9_+-]*\s*", "", text.strip())
        text = re.sub(r"\s*```\s*$", "", text)
        try:
            tree = ast.parse(text)
        except Exception:
            return set()
    roots: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                roots.add(node.module.split(".")[0])
    return roots


def normalize_backend_profile(profile: str | None) -> str:
    key = str(profile or "e2b").strip().lower()
    return _BACKEND_ALIASES.get(key, "e2b")


def normalize_required_dependencies(required_dependencies: Iterable[str] | None = None) -> Set[str]:
    return {
        str(dep).strip().split(".")[0].lower()
        for dep in (required_dependencies or [])
        if str(dep).strip()
    }


def requires_cloudrun_backend(required_dependencies: Iterable[str] | None = None) -> bool:
    required = normalize_required_dependencies(required_dependencies)
    return bool(required & set(CLOUDRUN_NATIVE_ALLOWLIST))


def cloudrun_imports_from_code(code: str) -> List[str]:
    imports = {imp.strip().lower() for imp in extract_import_roots(code) if str(imp).strip()}
    return sorted(imports & set(CLOUDRUN_NATIVE_ALLOWLIST))


def code_requires_cloudrun_backend(code: str) -> bool:
    return bool(cloudrun_imports_from_code(code))


def _allowed_import_roots(required: Set[str], backend_profile: str) -> Set[str]:
    stdlib = _stdlib_modules()
    allowed = set(BASE_ALLOWLIST) | (set(EXTENDED_ALLOWLIST) & required) | stdlib
    if backend_profile == "cloudrun":
        allowed |= set(CLOUDRUN_NATIVE_ALLOWLIST)
        allowed |= set(CLOUDRUN_OPTIONAL_ALLOWLIST)
    return allowed


def _banned_import_roots(backend_profile: str) -> Set[str]:
    banned = set(BANNED_ALWAYS_ALLOWLIST)
    if backend_profile != "cloudrun":
        banned |= set(E2B_HEAVY_BLOCKLIST)
    return banned


def check_dependency_precheck(
    code: str,
    required_dependencies: Iterable[str] | None = None,
    backend_profile: str = "e2b",
) -> Dict[str, List[str] | Dict[str, str] | str]:
    imports = {str(imp).strip().lower() for imp in extract_import_roots(code) if str(imp).strip()}
    required = normalize_required_dependencies(required_dependencies)
    backend = normalize_backend_profile(backend_profile)

    banned = sorted(imports & _banned_import_roots(backend))
    allowed = _allowed_import_roots(required, backend)
    blocked = sorted([imp for imp in imports if imp not in allowed])
    suggestions: Dict[str, str] = {}
    for imp in blocked + banned:
        if imp in {"pulp", "cvxpy"}:
            suggestions[imp] = "Use scipy.optimize.linprog or scipy.optimize.minimize (SLSQP)."
        elif imp in {"fuzzywuzzy"}:
            suggestions[imp] = "Use difflib or rapidfuzz (only if contract allows)."
        elif imp in {"rapidfuzz"}:
            suggestions[imp] = "Request rapidfuzz in execution_contract.required_dependencies, or use difflib."
        elif imp in {
            "torch",
            "transformers",
            "datasets",
            "tokenizers",
            "accelerate",
            "sentence_transformers",
            "xgboost",
            "lightgbm",
            "catboost",
            "shap",
            "optuna",
            "imblearn",
            "category_encoders",
            "plotly",
            "rapidfuzz",
            "pydantic",
            "pandera",
            "networkx",
            "sentencepiece",
            "huggingface_hub",
        }:
            if backend == "cloudrun":
                suggestions[imp] = (
                    "Use Cloud Run heavy runtime and declare dependency in execution_contract.required_dependencies."
                )
            else:
                suggestions[imp] = "Use Cloud Run heavy runtime or switch to scikit-learn/statsmodels."
        elif imp in {"tensorflow", "keras"}:
            suggestions[imp] = "TensorFlow/Keras are not enabled in current sandbox profiles."
        elif imp in {"spacy"}:
            suggestions[imp] = "Use standard NLP with scikit-learn or regex."
        elif imp in {"prophet"}:
            suggestions[imp] = "Use statsmodels or sklearn time-series approaches."
        elif imp in {"pyspark"}:
            suggestions[imp] = "Use pandas/pyarrow/duckdb for local processing."
    return {
        "imports": sorted(imports),
        "blocked": blocked,
        "banned": banned,
        "suggestions": suggestions,
        "backend_profile": backend,
    }


def get_sandbox_install_packages(
    required_dependencies: Iterable[str] | None = None,
    backend_profile: str = "e2b",
) -> Dict[str, List[str]]:
    required = normalize_required_dependencies(required_dependencies)
    backend = normalize_backend_profile(backend_profile)
    extra = [PIP_EXTENDED[d] for d in EXTENDED_ALLOWLIST if d in required]
    heavy = []
    if backend == "cloudrun":
        heavy = [
            PIP_CLOUDRUN_NATIVE[d]
            for d in CLOUDRUN_NATIVE_ALLOWLIST
            if d in required and d in PIP_CLOUDRUN_NATIVE
        ]
        heavy += [
            PIP_CLOUDRUN_OPTIONAL[d]
            for d in CLOUDRUN_OPTIONAL_ALLOWLIST
            if d in required and d in PIP_CLOUDRUN_OPTIONAL
        ]
        # Allow explicit request for joblib even though it is part of base dependencies.
        if "joblib" in required:
            heavy.append("joblib")
        # Deduplicate while preserving order.
        deduped: List[str] = []
        seen: Set[str] = set()
        for pkg in heavy:
            if pkg in seen:
                continue
            seen.add(pkg)
            deduped.append(pkg)
        heavy = deduped
    return {"base": list(PIP_BASE), "extra": extra, "heavy": heavy}
