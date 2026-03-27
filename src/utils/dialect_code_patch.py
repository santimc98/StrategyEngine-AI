"""
AST-based autopatcher for pd.read_csv dialect parameters.

This module provides deterministic, minimal patching of Python code to ensure
pd.read_csv calls include the correct sep/decimal/encoding parameters.

NOT a heuristic or dataset-specific fix - purely mechanical AST transformation.
"""

import ast
import copy
from typing import List, Tuple


_CSV_DIALECT_HELPER_NAME = "_strip_csv_dialect_kwargs"
_CSV_DIALECT_HELPER_SOURCE = f"""
def {_CSV_DIALECT_HELPER_NAME}(kwargs):
    if kwargs is None:
        return {{}}
    if not hasattr(kwargs, "items"):
        return kwargs
    return {{
        key: value
        for key, value in kwargs.items()
        if str(key) not in {{"sep", "delimiter", "decimal", "encoding"}}
    }}
""".strip()


def patch_read_csv_dialect(
    code: str,
    csv_sep: str,
    csv_decimal: str,
    csv_encoding: str,
    expected_path: str = "data/raw.csv",
) -> Tuple[str, List[str], bool]:
    """
    Autopatch pd.read_csv to include correct dialect parameters.

    This function:
    1. Parses code as AST
    2. Finds the target read_csv call (one reading expected_path, or first one)
    3. For that call:
       - If sep/decimal/encoding is missing, adds it
       - If sep/decimal/encoding (or delimiter alias) is a literal with wrong value, replaces it
       - Does NOT touch non-literal values (variables/expressions)
       - If **kwargs is present, strips dialect keys from it and pins explicit dialect params

    Args:
        code: Python source code to patch
        csv_sep: Expected separator (e.g., ",", ";", "\\t")
        csv_decimal: Expected decimal (e.g., ".", ",")
        csv_encoding: Expected encoding (e.g., "utf-8")
        expected_path: Path to match for target read_csv selection

    Returns:
        Tuple of (patched_code, patch_notes, changed_flag):
        - patched_code: The patched source (or original if no changes/error)
        - patch_notes: List of notes describing what was patched
        - changed_flag: True if any changes were made
    """
    patch_notes: List[str] = []

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return code, [f"AST parse error: {e}"], False

    changed = False
    helper_needed = False

    class _DialectAliasLookupTransformer(ast.NodeTransformer):
        def __init__(self) -> None:
            self.changed = False

        def visit_Call(self, node: ast.Call) -> ast.AST:
            node = self.generic_visit(node)
            func = getattr(node, "func", None)
            if not isinstance(func, ast.Attribute) or func.attr != "get":
                return node
            if not node.args:
                return node
            first_arg = node.args[0]
            if not (isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str)):
                return node
            key = first_arg.value
            if key not in {"sep", "delimiter"}:
                return node

            default_arg = copy.deepcopy(node.args[1]) if len(node.args) > 1 else ast.Constant(value=None)
            owner = copy.deepcopy(func.value)
            normalized = ast.BoolOp(
                op=ast.Or(),
                values=[
                    ast.Call(
                        func=ast.Attribute(value=copy.deepcopy(owner), attr="get", ctx=ast.Load()),
                        args=[ast.Constant(value="sep")],
                        keywords=[],
                    ),
                    ast.Call(
                        func=ast.Attribute(value=copy.deepcopy(owner), attr="get", ctx=ast.Load()),
                        args=[ast.Constant(value="delimiter"), default_arg],
                        keywords=[],
                    ),
                ],
            )
            self.changed = True
            return ast.copy_location(normalized, node)

    alias_transformer = _DialectAliasLookupTransformer()
    tree = alias_transformer.visit(tree)
    if alias_transformer.changed:
        patch_notes.append("Normalized dialect.get sep/delimiter alias lookups")
        changed = True

    # Find all pd.read_csv calls
    calls: List[ast.Call] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "read_csv":
                if isinstance(func.value, ast.Name) and func.value.id == "pd":
                    calls.append(node)
            elif isinstance(func, ast.Name) and func.id == "read_csv":
                calls.append(node)

    if not calls:
        if not changed:
            return code, ["No pd.read_csv calls found"], False
        try:
            ast.fix_missing_locations(tree)
            patched_code = ast.unparse(tree)
        except Exception as e:
            return code, [f"AST unparse error: {e}"], False
        return patched_code, patch_notes, True

    # Select target call: prefer one reading expected_path, else first
    def _is_expected_path(arg: ast.AST) -> bool:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value == expected_path
        return False

    target_call = None
    for call in calls:
        if call.args and _is_expected_path(call.args[0]):
            target_call = call
            break
    if target_call is None:
        target_call = calls[0]

    # Expected parameters
    expected = {
        "sep": csv_sep,
        "decimal": csv_decimal,
        "encoding": csv_encoding,
    }

    def _normalize_encoding(value: str) -> str:
        return str(value).strip().lower().replace("_", "-")

    def _encoding_matches(expected_value: str, actual_value: str) -> bool:
        exp = _normalize_encoding(expected_value)
        act = _normalize_encoding(actual_value)
        if exp in {"utf-8", "utf8"}:
            return act in {"utf-8", "utf8", "utf-8-sig", "utf8-sig"}
        if exp in {"utf-8-sig", "utf8-sig"}:
            return act in {"utf-8", "utf8", "utf-8-sig", "utf8-sig"}
        return exp == act

    def _is_helper_call(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == _CSV_DIALECT_HELPER_NAME
            and len(node.args) == 1
        )

    def _helper_already_defined(module_tree: ast.Module) -> bool:
        for stmt in getattr(module_tree, "body", []):
            if isinstance(stmt, ast.FunctionDef) and stmt.name == _CSV_DIALECT_HELPER_NAME:
                return True
        return False

    named_keywords = [kw for kw in target_call.keywords if kw.arg is not None]
    star_keywords = [kw for kw in target_call.keywords if kw.arg is None]
    has_kwargs = bool(star_keywords)

    if has_kwargs:
        sanitized_any = False
        for kw in star_keywords:
            if _is_helper_call(kw.value):
                continue
            kw.value = ast.Call(
                func=ast.Name(id=_CSV_DIALECT_HELPER_NAME, ctx=ast.Load()),
                args=[kw.value],
                keywords=[],
            )
            sanitized_any = True
        if sanitized_any:
            patch_notes.append("Sanitized **kwargs to strip sep/delimiter/decimal/encoding before pd.read_csv")
            changed = True
            helper_needed = True

        named_without_alias = [kw for kw in named_keywords if kw.arg != "delimiter"]
        kw_map = {kw.arg: kw for kw in named_without_alias if kw.arg is not None}
        delimiter_kw = next((kw for kw in named_keywords if kw.arg == "delimiter"), None)
        if delimiter_kw is not None:
            patch_notes.append("Normalized delimiter alias to explicit sep= for pd.read_csv")
            changed = True

        for param, expected_value in expected.items():
            kw = kw_map.get(param)
            if kw is None:
                named_without_alias.append(
                    ast.keyword(arg=param, value=ast.Constant(value=expected_value))
                )
                patch_notes.append(f"Added {param}={repr(expected_value)}")
                changed = True
            else:
                literal_matches = (
                    isinstance(kw.value, ast.Constant)
                    and isinstance(kw.value.value, str)
                    and (
                        _encoding_matches(expected_value, kw.value.value)
                        if param == "encoding"
                        else kw.value.value == expected_value
                    )
                )
                if not literal_matches:
                    kw.value = ast.Constant(value=expected_value)
                    patch_notes.append(f"Pinned {param}={repr(expected_value)} ahead of **kwargs")
                    changed = True

        target_call.keywords = named_without_alias + star_keywords
    else:
        kw_map = {kw.arg: kw for kw in named_keywords if kw.arg is not None}
        delimiter_kw = kw_map.get("delimiter")

        for param, expected_value in expected.items():
            alias_kw = delimiter_kw if param == "sep" else None
            kw = kw_map.get(param) or alias_kw
            if kw is None:
                target_call.keywords.append(
                    ast.keyword(arg=param, value=ast.Constant(value=expected_value))
                )
                patch_notes.append(f"Added {param}={repr(expected_value)}")
                changed = True
                continue

            val_node = kw.value
            if isinstance(val_node, ast.Constant) and isinstance(val_node.value, str):
                actual_value = val_node.value
                matches = (
                    _encoding_matches(expected_value, actual_value)
                    if param == "encoding"
                    else actual_value == expected_value
                )
                if not matches:
                    kw.value = ast.Constant(value=expected_value)
                    label = "delimiter" if kw.arg == "delimiter" else param
                    patch_notes.append(
                        f"Replaced {label}={repr(actual_value)} with {repr(expected_value)}"
                    )
                    changed = True

    if not changed:
        return code, patch_notes or ["No changes needed"], False

    if helper_needed and not _helper_already_defined(tree):
        helper_def = ast.parse(_CSV_DIALECT_HELPER_SOURCE).body[0]
        insert_at = 0
        body = list(tree.body or [])
        if body:
            first = body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(getattr(first, "value", None), ast.Constant)
                and isinstance(first.value.value, str)
            ):
                insert_at = 1
            while insert_at < len(body):
                stmt = body[insert_at]
                if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                    insert_at += 1
                    continue
                break
        tree.body.insert(insert_at, helper_def)
        changed = True

    # Re-serialize the AST
    try:
        ast.fix_missing_locations(tree)
        patched_code = ast.unparse(tree)
    except Exception as e:
        return code, [f"AST unparse error: {e}"], False

    return patched_code, patch_notes, True


def has_kwargs_in_read_csv(code: str, expected_path: str = "data/raw.csv") -> bool:
    """
    Check if the target read_csv call has **kwargs.

    Args:
        code: Python source code
        expected_path: Path to match for target read_csv selection

    Returns:
        True if target read_csv has **kwargs (keyword with arg=None)
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    calls: List[ast.Call] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "read_csv":
                if isinstance(func.value, ast.Name) and func.value.id == "pd":
                    calls.append(node)
            elif isinstance(func, ast.Name) and func.id == "read_csv":
                calls.append(node)

    if not calls:
        return False

    def _is_expected_path(arg: ast.AST) -> bool:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value == expected_path
        return False

    target_call = None
    for call in calls:
        if call.args and _is_expected_path(call.args[0]):
            target_call = call
            break
    if target_call is None:
        target_call = calls[0]

    return any(kw.arg is None for kw in target_call.keywords)
