from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools" / "wids_2026_prep" / "build_wids_2026_unified.py"


def test_build_wids_2026_unified_with_metadata_and_sample(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    train_df = pd.DataFrame(
        {
            "event_id": [101, 102],
            "feature_a": [1.5, 2.5],
            "time_to_hit_hours": [10.0, 40.0],
            "event": [1, 1],
        }
    )
    test_df = pd.DataFrame(
        {
            "event_id": [201],
            "feature_a": [3.5],
        }
    )
    metadata_df = pd.DataFrame(
        {
            "event_id": [101, 102, 201],
            "region": ["north", "south", "west"],
        }
    )
    sample_df = pd.DataFrame(
        {
            "event_id": [201],
            "prob_12h": [0.0],
            "prob_24h": [0.0],
            "prob_48h": [0.0],
            "prob_72h": [0.0],
        }
    )

    train_df.to_csv(input_dir / "train.csv", index=False)
    test_df.to_csv(input_dir / "test.csv", index=False)
    metadata_df.to_csv(input_dir / "metadata.csv", index=False)
    sample_df.to_csv(input_dir / "sample_submission.csv", index=False)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout

    unified_df = pd.read_csv(output_dir / "wids_2026_unified.csv")
    summary = json.loads((output_dir / "prep_summary.json").read_text(encoding="utf-8"))

    assert "__split" in unified_df.columns
    assert "label_12h" in unified_df.columns
    assert "label_24h" in unified_df.columns
    assert "label_48h" in unified_df.columns
    assert "label_72h" in unified_df.columns
    assert "region" in unified_df.columns
    assert unified_df["__split"].tolist() == ["train", "train", "test"]

    first_train = unified_df.loc[unified_df["event_id"] == 101].iloc[0]
    second_train = unified_df.loc[unified_df["event_id"] == 102].iloc[0]
    test_row = unified_df.loc[unified_df["event_id"] == 201].iloc[0]

    assert int(first_train["label_12h"]) == 1
    assert int(first_train["label_24h"]) == 1
    assert int(second_train["label_12h"]) == 0
    assert int(second_train["label_48h"]) == 1
    assert pd.isna(test_row["label_12h"])

    assert summary["sample_submission_check"]["same_ids"] is True
    assert summary["sample_submission_check"]["same_order"] is True


def test_build_wids_2026_unified_accepts_column_dictionary_metadata(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    pd.DataFrame(
        {
            "event_id": [1],
            "feature_x": [2.0],
            "time_to_hit_hours": [14.0],
            "event": [1],
        }
    ).to_csv(input_dir / "train.csv", index=False)
    pd.DataFrame(
        {
            "event_id": [2],
            "feature_x": [3.0],
        }
    ).to_csv(input_dir / "test.csv", index=False)
    pd.DataFrame(
        {
            "column": ["event_id", "feature_x", "time_to_hit_hours", "event"],
            "type": ["identifier", "feature", "target", "target"],
            "description": ["id", "feature", "time", "event flag"],
        }
    ).to_csv(input_dir / "metaData.csv", index=False)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout

    summary = json.loads((output_dir / "prep_summary.json").read_text(encoding="utf-8"))
    metadata_summary = json.loads((output_dir / "column_metadata_summary.json").read_text(encoding="utf-8"))

    assert summary["metadata_mode"] == "column_dictionary"
    assert summary["metadata_merged"] is False
    assert metadata_summary["mode"] == "column_dictionary"
    assert "event_id" in metadata_summary["declared_columns_present"]
