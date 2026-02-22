from src.agents.execution_planner import _compress_text_preserve_ends, MINIMAL_CONTRACT_COMPILER_PROMPT


def test_compress_text_preserve_ends_keeps_tail():
    head = "HEAD"
    tail = "Tail KPI Accuracy"
    middle = "x" * 200
    text = head + middle + tail
    compressed = _compress_text_preserve_ends(text, max_chars=60, head=20, tail=20)
    assert "HEAD" in compressed
    assert "Accuracy" in compressed
    assert "..." in compressed
    assert len(compressed) <= 60 + len("\n...\n")


def test_minimal_contract_prompt_defines_column_roles_semantics():
    prompt = MINIMAL_CONTRACT_COMPILER_PROMPT
    assert "Role definitions (ML execution context)" in prompt
    assert "outcome MUST contain only the target" in prompt


def test_minimal_contract_prompt_declares_phased_compilation():
    prompt = MINIMAL_CONTRACT_COMPILER_PROMPT
    assert "Phased contract compilation protocol" in prompt
    assert "Phase 1 FACTS_EXTRACTOR" in prompt
    assert "Phase 4 VALIDATOR_REPAIR" in prompt
