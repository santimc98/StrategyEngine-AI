from src.graph import graph as graph_module
from src.utils.run_bundle import init_run_bundle


class StubDataEngineer:
    def __init__(self) -> None:
        self.calls = []
        self.model_name = "stub"
        self.last_prompt = None
        self.last_response = None

    def generate_cleaning_script(
        self,
        data_audit,
        strategy,
        input_path,
        business_objective="",
        csv_encoding="utf-8",
        csv_sep=",",
        csv_decimal=".",
        execution_contract=None,
        de_view=None,
        repair_mode=False,
    ) -> str:
        self.calls.append(data_audit)
        if len(self.calls) == 1:
            self.last_response = "# Error: Data Engineer Failed: INVALID_PYTHON_SYNTAX"
            return self.last_response
        self.last_response = "# Error: stop"
        return self.last_response


class StubDataEngineerPromptPath:
    def __init__(self) -> None:
        self.calls = []
        self.model_name = "stub"
        self.last_prompt = None
        self.last_response = None

    def generate_cleaning_script(
        self,
        data_audit,
        strategy,
        input_path,
        prompt_input_path=None,
        business_objective="",
        csv_encoding="utf-8",
        csv_sep=",",
        csv_decimal=".",
        execution_contract=None,
        de_view=None,
        repair_mode=False,
    ) -> str:
        self.calls.append(
            {
                "input_path": input_path,
                "prompt_input_path": prompt_input_path,
            }
        )
        self.last_response = "# Error: stop"
        return self.last_response


def test_generation_failure_retry_includes_error_context(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv_path = tmp_path / "input.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")
    state = {
        "csv_path": str(csv_path),
        "selected_strategy": {"required_columns": []},
        "business_objective": "test",
        "csv_encoding": "utf-8",
        "csv_sep": ",",
        "csv_decimal": ".",
        "data_summary": "summary",
        "execution_contract": {"required_outputs": ["data/cleaned_data.csv"]},
        "de_view": {
            "output_path": "data/cleaned_data.csv",
            "output_manifest_path": "data/cleaning_manifest.json",
            "required_columns": [],
            "cleaning_gates": [],
            "data_engineer_runbook": {"steps": ["load", "clean", "persist"]},
        },
        "run_id": "testrun-generation-retry",
        "run_start_epoch": 0,
    }
    run_dir = tmp_path / "runs" / state["run_id"]
    init_run_bundle(state["run_id"], state, run_dir=str(run_dir))

    stub = StubDataEngineer()
    monkeypatch.setattr(graph_module, "data_engineer", stub)

    result = graph_module.run_data_engineer(state)

    assert len(stub.calls) == 2
    # The retry context now uses ITERATION_FEEDBACK_CONTEXT with structured JSON
    # that includes the generation failure details
    retry_context = stub.calls[1]
    assert "ITERATION_FEEDBACK" in retry_context or "GENERATION_FAILURE" in retry_context
    assert "INVALID_PYTHON_SYNTAX" in retry_context
    assert result.get("pipeline_aborted_reason") == "data_engineer_generation_failed"


def test_run_data_engineer_passes_local_csv_for_prompt_profiling(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv_path = tmp_path / "input.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")
    state = {
        "csv_path": str(csv_path),
        "selected_strategy": {"required_columns": []},
        "business_objective": "test",
        "csv_encoding": "utf-8",
        "csv_sep": ",",
        "csv_decimal": ".",
        "data_summary": "summary",
        "execution_contract": {"required_outputs": ["data/cleaned_data.csv"]},
        "de_view": {
            "output_path": "data/cleaned_data.csv",
            "output_manifest_path": "data/cleaning_manifest.json",
            "required_columns": [],
            "cleaning_gates": [],
            "data_engineer_runbook": {"steps": ["load", "clean", "persist"]},
        },
        "run_id": "testrun-prompt-input-path",
        "run_start_epoch": 0,
    }
    run_dir = tmp_path / "runs" / state["run_id"]
    init_run_bundle(state["run_id"], state, run_dir=str(run_dir))

    stub = StubDataEngineerPromptPath()
    monkeypatch.setattr(graph_module, "data_engineer", stub)

    result = graph_module.run_data_engineer(state)

    assert stub.calls
    assert stub.calls[0]["input_path"] == "data/raw.csv"
    assert stub.calls[0]["prompt_input_path"] == str(csv_path)
    assert result.get("pipeline_aborted_reason") == "data_engineer_generation_failed"
