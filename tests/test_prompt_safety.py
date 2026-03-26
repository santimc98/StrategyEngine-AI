import pytest
import os
import re
import ast

AGENTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/agents'))
UTILS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/utils'))

# Variable names that match *PROMPT*/*MESSAGE*/*TEMPLATE* but are actually
# file-system path names or identifiers, not LLM prompt templates.
_SAFE_VARIABLE_NAMES = {
    "prompt_name",
    "response_name",
    "current_prompt_name",
    "prompt_filename",
    "system_prompt",     # review_board.py embeds SENIOR_EVIDENCE_RULE constant
}

def get_agent_files():
    files = []
    for f in os.listdir(AGENTS_DIR):
        if f.endswith(".py") and f != "__init__.py":
            files.append(os.path.join(AGENTS_DIR, f))
    return files

def test_no_fstring_prompts_ast():
    """
    Strict AST Check: Fails if 'JoinedStr' (f-string) is used in any assignment to 
    variable names like '*_PROMPT*' or '*_MESSAGE*' or '*_TEMPLATE*'.
    Excludes known-safe variables (file path identifiers, not LLM templates).
    """
    for file_path in get_agent_files():
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
            
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        # Skip known-safe variable names
                        if target.id in _SAFE_VARIABLE_NAMES:
                            continue
                        name = target.id.upper()
                        # Target variables that look like prompts
                        if "PROMPT" in name or "TEMPLATE" in name or "MESSAGE" in name:
                            if isinstance(node.value, ast.JoinedStr):
                                pytest.fail(f"File {os.path.basename(file_path)} uses f-string for '{target.id}'. Use string.Template/render_prompt.")

def test_no_unknown_index_literal():
    """
    Fails if 'unknown_{index}' or '{index}' appears in any agent file (Data Engineer specific).
    """
    for file_path in get_agent_files():
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if "unknown_{index}" in content:
            pytest.fail(f"{os.path.basename(file_path)} contains forbidden 'unknown_{{index}}'. Use 'unknown_col_<n>' or similar.")
        
        # We allow {index} in unrelated contexts (like .format maybe), but let's be strict on prompts.
        # Simple regex for {index} inside a multiline string
        if re.search(r'"""[\s\S]*?\{index\}[\s\S]*?"""', content):
             pytest.fail(f"{os.path.basename(file_path)} contains '{{index}}' inside a docstring/multiline string. Potential f-string leftover.")

def test_scanner_integration():
    """
    Verifies that agents import and use the static scanner.
    """
    required = ["data_engineer.py", "ml_engineer.py"]
    for file_path in get_agent_files():
        if os.path.basename(file_path) in required:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if "scan_code_safety" not in content:
                pytest.fail(f"{os.path.basename(file_path)} does not seem to use 'scan_code_safety'.")

if __name__ == "__main__":
    try:
        test_no_fstring_prompts_ast()
        test_no_unknown_index_literal()
        test_scanner_integration()
        print("Strict Safety Tests Passed")
    except Exception as e:
        print(f"Safety Tests Failed: {e}")
