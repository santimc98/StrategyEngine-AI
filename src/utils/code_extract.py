import re
import ast
import json


def extract_code_block(text: str) -> str:
    """
    Extracts code-like content from LLM output.
    - If complete fenced blocks exist, returns their concatenation.
    - If an unterminated/single fence exists, chooses the most structured side
      (prefix/suffix) instead of always taking suffix.
    - If the response is bare (no fence) but contains narrative/reasoning text
      before the actual code, strips the prefix until a valid Python parse begins.
    - Otherwise returns trimmed text.
    """
    if not isinstance(text, str):
        return str(text)
    # Prefer all fenced code blocks (python or generic) and join.
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    cleaned = [b.strip() for b in blocks if isinstance(b, str) and b.strip()]
    if cleaned:
        return "\n\n".join(cleaned).strip()
    # Handle unterminated / single fences (e.g., truncated output or stray fence).
    m = re.search(r"```(?:python)?", text, re.IGNORECASE)
    if m:
        prefix = text[:m.start()].strip()
        suffix = text[m.end():].strip()
        candidates = [c for c in (prefix, suffix) if isinstance(c, str) and c.strip()]
        if not candidates:
            return ""

        def _score(candidate: str) -> float:
            s = candidate.strip()
            score = 0.0
            if is_syntax_valid(s):
                score += 100.0
            try:
                json.loads(s)
                score += 90.0
            except Exception:
                pass
            lowered = s.lower()
            for marker in ("import ", "from ", "def ", "class ", "if __name__", " = ", "print(", "{", "}", "[", "]", "\""):
                score += 1.0 * lowered.count(marker)
            score += min(len(s), 12000) / 12000.0
            return score

        return max(candidates, key=_score)
    # No fence present. If the whole text parses, return it as-is.
    stripped = text.strip()
    if not stripped:
        return ""
    if is_syntax_valid(stripped):
        return stripped
    # Otherwise the response likely contains a narrative/reasoning prefix
    # followed by the real code (a common LLM violation of "code only" prompts).
    # Locate the earliest line from which the remaining text parses cleanly.
    recovered = _strip_leading_non_code_prefix(stripped)
    if recovered is not None:
        return recovered
    return stripped


_PYTHON_START_TOKENS = (
    "import ", "from ", "def ", "class ", "async def ",
    "@", "if __name__", "if ", "for ", "while ", "try:",
    "with ", "#", '"""', "'''",
)


def _strip_leading_non_code_prefix(text: str) -> str | None:
    """Return the largest suffix of `text` that parses as valid Python.

    Scans forward line-by-line from the top, skipping likely-prose lines, and
    tries to parse what remains. Returns None if no parseable suffix is found.

    Universal by design: does not whitelist specific prose patterns. It only
    tests whether the substring starting at each line is syntactically valid
    Python, so it recovers from any narrative prefix an LLM might emit.
    """
    lines = text.split("\n")
    n = len(lines)
    if n == 0:
        return None
    # Fast path: start from lines that look like Python entry points.
    candidate_starts: list[int] = []
    for idx, raw in enumerate(lines):
        stripped_line = raw.lstrip()
        if not stripped_line:
            continue
        if stripped_line.startswith(_PYTHON_START_TOKENS):
            candidate_starts.append(idx)
    # Also allow brute-force scan up to a sensible cap to avoid quadratic cost.
    brute_limit = min(n, 400)
    brute_starts = [i for i in range(brute_limit) if i not in set(candidate_starts)]
    # Prefer earliest Python-like start that parses; fall back to brute scan.
    for idx in candidate_starts + brute_starts:
        candidate = "\n".join(lines[idx:]).strip()
        if not candidate:
            continue
        if is_syntax_valid(candidate):
            return candidate
    return None


def is_syntax_valid(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False
