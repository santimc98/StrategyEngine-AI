# WiDS 2026 Prep

Put the competition files here:

- `input/train.csv`
- `input/test.csv`
- `input/metadata.csv` or `input/metaData.csv` (optional)
- `input/sample_submission.csv` (optional but recommended)

Run from the repo root:

```powershell
python tools/wids_2026_prep/build_wids_2026_unified.py
```

Outputs:

- `tools/wids_2026_prep/output/wids_2026_unified.csv`
- `tools/wids_2026_prep/output/prep_summary.json`

The generated CSV is optimized for the current agent stack:

- explicit `__split` column
- flattened tabular columns
- train-only labels `label_12h`, `label_24h`, `label_48h`, `label_72h`
- optional metadata merged on `event_id`
- optional validation against `sample_submission.csv`

If the metadata file is a column dictionary instead of row-level metadata, the script will not merge it into the rows. It will write `output/column_metadata_summary.json` instead.

If your raw files use different names for the target columns, pass them explicitly:

```powershell
python tools/wids_2026_prep/build_wids_2026_unified.py --time-column time_to_hit_hours --event-column event
```
