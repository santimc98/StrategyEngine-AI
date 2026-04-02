import pandas as pd
import numpy as np
import os
import csv
import re
import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv
from openai import OpenAI

try:
    import google.generativeai as genai  # type: ignore
except Exception:
    class _GenAIShim:
        GenerativeModel = None

    genai = _GenAIShim()

load_dotenv()
from src.utils.pii_scrubber import PIIScrubber


# =============================================================================
# SMART SAMPLING CONFIGURATION
# =============================================================================
# Instead of reading only the first N rows (biased for time-sorted data),
# we use a composite sampling strategy: Head + Tail + Random Middle
# This captures temporal/distributional range for better profiling.

_MIN_SAMPLE_SIZE = 3000      # Minimum total sample
_MAX_SAMPLE_SIZE = 15000     # Safety cap to protect memory
_HEAD_RATIO = 0.4            # 40% from head
_TAIL_RATIO = 0.4            # 40% from tail
_RANDOM_RATIO = 0.2          # 20% from random middle (if file supports it)
_FILE_SIZE_THRESHOLD_MB = 10 # Only sample if file > this size


def _compute_sample_sizes(file_size_mb: float) -> Dict[str, int]:
    """
    Compute context-aware sample sizes based on file size.

    Returns dict with 'head', 'tail', 'random', 'total' sample sizes.
    """
    # Scale sample size with file size, but cap it
    if file_size_mb <= _FILE_SIZE_THRESHOLD_MB:
        # Small file - no sampling needed
        return {"head": 0, "tail": 0, "random": 0, "total": 0, "strategy": "full_read"}

    # Base sample size scales with file size (logarithmic scaling)
    # e.g., 10MB -> 5000, 100MB -> 8000, 1GB -> 12000
    import math
    base_sample = int(5000 + 2000 * math.log10(max(1, file_size_mb / 10)))
    total_sample = min(max(base_sample, _MIN_SAMPLE_SIZE), _MAX_SAMPLE_SIZE)

    head_size = int(total_sample * _HEAD_RATIO)
    tail_size = int(total_sample * _TAIL_RATIO)
    random_size = total_sample - head_size - tail_size  # Remainder for random

    return {
        "head": head_size,
        "tail": tail_size,
        "random": random_size,
        "total": total_sample,
        "strategy": "composite_head_tail_random"
    }


def _read_csv_composite_sample(
    data_path: str,
    sep: str,
    decimal: str,
    encoding: str,
    file_size_mb: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Read CSV using composite sampling strategy (Head + Tail + Random).

    This captures both the beginning and end of the data, which is crucial
    for time-sorted or grouped datasets where the first N rows may only
    represent a small slice of the distribution.

    Returns:
        Tuple of (DataFrame, sampling_metadata)
    """
    sample_config = _compute_sample_sizes(file_size_mb)

    if sample_config["strategy"] == "full_read":
        # Small file - read everything
        df = pd.read_csv(data_path, sep=sep, decimal=decimal, encoding=encoding)
        return df, {
            "was_sampled": False,
            "strategy": "full_read",
            "total_rows_read": len(df),
        }

    head_size = sample_config["head"]
    tail_size = sample_config["tail"]
    random_size = sample_config["random"]
    total_target = sample_config["total"]

    print(f"Steward: Composite sampling - Head={head_size}, Tail={tail_size}, Random={random_size} (File: {file_size_mb:.2f}MB)")

    # Step 1: Read HEAD rows
    df_head = pd.read_csv(
        data_path, sep=sep, decimal=decimal, encoding=encoding,
        nrows=head_size, dtype=str, low_memory=False
    )

    # Step 2: Count total rows (efficient - just count lines)
    total_rows = _count_csv_rows(data_path, encoding)

    # Step 3: Read TAIL rows (skip to end)
    df_tail = pd.DataFrame()
    if total_rows > head_size + tail_size:
        skip_rows = max(0, total_rows - tail_size)
        # Skip header + rows to get to tail
        try:
            df_tail = pd.read_csv(
                data_path, sep=sep, decimal=decimal, encoding=encoding,
                skiprows=range(1, skip_rows + 1),  # +1 to skip header row index
                dtype=str, low_memory=False
            )
        except Exception as e:
            print(f"Steward: Tail sampling failed ({e}), using head only")
            df_tail = pd.DataFrame()
    elif total_rows > head_size:
        # File is small enough that head overlaps with tail
        # Read everything after head
        try:
            df_tail = pd.read_csv(
                data_path, sep=sep, decimal=decimal, encoding=encoding,
                skiprows=range(1, head_size + 1),
                dtype=str, low_memory=False
            )
        except Exception:
            df_tail = pd.DataFrame()

    # Step 4: Random middle sample (probabilistic skip if file is large)
    df_random = pd.DataFrame()
    if random_size > 0 and total_rows > head_size + tail_size + random_size:
        try:
            # Calculate rows to sample from middle section
            middle_start = head_size
            middle_end = total_rows - tail_size
            middle_size = middle_end - middle_start

            if middle_size > random_size:
                # Probabilistic sampling from middle
                # Sample random row indices from middle section
                np.random.seed(42)  # Reproducible
                random_indices = sorted(np.random.choice(
                    range(middle_start, middle_end),
                    size=min(random_size, middle_size),
                    replace=False
                ))

                # Read only those rows (skip all others)
                # This is more efficient than reading all and sampling
                skip_set = set(range(1, total_rows + 1)) - set(idx + 1 for idx in random_indices)
                df_random = pd.read_csv(
                    data_path, sep=sep, decimal=decimal, encoding=encoding,
                    skiprows=list(skip_set)[:total_rows],  # Limit skip list
                    dtype=str, low_memory=False, nrows=random_size
                )
        except Exception as e:
            print(f"Steward: Random middle sampling failed ({e}), using head+tail only")
            df_random = pd.DataFrame()

    # Step 5: Combine samples
    dfs_to_concat = [df_head]
    if not df_tail.empty:
        dfs_to_concat.append(df_tail)
    if not df_random.empty:
        dfs_to_concat.append(df_random)

    df_combined = pd.concat(dfs_to_concat, ignore_index=True)

    # Step 6: Remove duplicates (in case tail overlaps with head)
    df_combined = df_combined.drop_duplicates()

    # Step 7: Shuffle for unbiased example rows
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

    actual_strategy = "head_tail_random" if not df_random.empty else "head_tail"

    sampling_metadata = {
        "was_sampled": True,
        "strategy": actual_strategy,
        "head_rows": len(df_head),
        "tail_rows": len(df_tail) if not df_tail.empty else 0,
        "random_rows": len(df_random) if not df_random.empty else 0,
        "total_rows_read": len(df_combined),
        "total_rows_in_file": total_rows,
        "shuffled": True,
    }

    print(f"Steward: Composite sample created - {len(df_combined)} rows from {total_rows} total (shuffled)")

    return df_combined, sampling_metadata


def _count_csv_rows(file_path: str, encoding: str) -> int:
    """Efficiently count rows in CSV file without loading into memory."""
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            # Count lines, subtract 1 for header
            row_count = sum(1 for _ in f) - 1
            return max(0, row_count)
    except Exception:
        # Fallback: estimate from file size (rough: ~100 bytes per row)
        file_size = os.path.getsize(file_path)
        return max(1000, file_size // 100)

class StewardAgent:
    def __init__(self, api_key: str = None):
        """
        Initializes the Steward Agent via the configured text provider.
        """
        self.provider = "openrouter"
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        google_api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model_name = os.getenv("STEWARD_MODEL", "google/gemini-3-flash-preview")
        self.client = None
        if self.api_key:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://openrouter.ai/api/v1",
            )
        elif google_api_key and getattr(genai, "GenerativeModel", None):
            self.provider = "gemini"
            try:
                configure = getattr(genai, "configure", None)
                if callable(configure):
                    configure(api_key=google_api_key)
            except Exception:
                pass
            self.client = genai.GenerativeModel(self.model_name)
        if self.client is None:
            raise ValueError("OPENROUTER_API_KEY or GOOGLE_API_KEY is required.")
        self.last_prompt = None
        self.last_response = None

    def _run_text_prompt(self, prompt: str) -> Tuple[str, Any, Any]:
        self.last_prompt = prompt
        if self.provider == "gemini" and hasattr(self.client, "generate_content"):
            response = self.client.generate_content(prompt)
            text = str(getattr(response, "text", "") or "").strip()
            finish_reason = getattr(response, "finish_reason", None)
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            choices = getattr(response, "choices", None) or []
            choice = choices[0] if choices else None
            message = getattr(choice, "message", None)
            text = getattr(message, "content", "") or ""
            text = str(text).strip()
            finish_reason = getattr(choice, "finish_reason", None) if choice is not None else None
        self.last_response = text
        return text, response, finish_reason

    def analyze_data(self, data_path: str, business_objective: str = "") -> Dict[str, Any]:
        """
        Analyzes the CSV file and generates a dense textual summary.
        Context-aware: audits based on the business_objective.
        Robustness V3: Implements automatic dialect detection and smart profiling.
        """
        # 1. Detect Encoding
        encodings = ['utf-8', 'latin-1', 'cp1252']
        detected_encoding = 'utf-8' # Default
        
        for enc in encodings:
            try:
                with open(data_path, 'r', encoding=enc) as f:
                    f.read(4096)
                detected_encoding = enc
                break
            except UnicodeDecodeError:
                continue
            except Exception:
                continue
        
        # 2. Detect Dialect (Robust V3)
        dialect_info = self._detect_csv_dialect(data_path, detected_encoding)
        sep = dialect_info['sep']
        decimal = dialect_info['decimal']
        print(f"Steward Detected: Sep='{sep}', Decimal='{decimal}', Encoding='{detected_encoding}'")

        try:
            # 3. Load Data with COMPOSITE SAMPLING (Head + Tail + Random)
            # This captures temporal/distributional range for better profiling
            file_size = os.path.getsize(data_path)
            file_size_mb = file_size / (1024 * 1024)

            # Primary Load Attempt with Smart Composite Sampling
            sampling_metadata = {}
            try:
                df, sampling_metadata = _read_csv_composite_sample(
                    data_path=data_path,
                    sep=sep,
                    decimal=decimal,
                    encoding=detected_encoding,
                    file_size_mb=file_size_mb,
                )
                was_sampled = sampling_metadata.get("was_sampled", False)
            except Exception as e:
                print(f"Steward: Composite sampling failed ({e}). Attempting fallback engine...")
                # Fallback: Python engine is slower but more robust
                df = pd.read_csv(data_path, sep=sep if sep else None, decimal=decimal,
                               encoding=detected_encoding, engine='python', on_bad_lines='skip')
                was_sampled = False
                sampling_metadata = {"was_sampled": False, "strategy": "fallback_full_read"}

            # 4. Preserve raw headers & Scrub
            df.columns = [str(c) for c in df.columns]
            pii_findings = detect_pii_findings(df)
            scrubber = PIIScrubber()
            df = scrubber.scrub_dataframe(df)

            # 5. Smart Profiling (V3)
            profile = self._smart_profile(df)
            shape = df.shape
            dataset_profile = build_dataset_profile(
                df=df,
                objective=business_objective,
                dialect_info=dialect_info,
                encoding=detected_encoding,
                file_size_bytes=file_size,
                was_sampled=was_sampled,
                sample_size=sampling_metadata.get("total_rows_read", shape[0]),
                pii_findings=pii_findings,
                sampling_metadata=sampling_metadata,  # Pass full sampling info
            )
            try:
                write_dataset_profile(dataset_profile)
            except Exception:
                pass

            # 6. Construct Prompt with TRANSPARENT sampling info
            sampling_strategy = sampling_metadata.get("strategy", "unknown")
            total_in_file = sampling_metadata.get("total_rows_in_file", "unknown")
            head_rows = sampling_metadata.get("head_rows", 0)
            tail_rows = sampling_metadata.get("tail_rows", 0)
            random_rows = sampling_metadata.get("random_rows", 0)

            if was_sampled:
                sampling_description = (
                    f"SAMPLING: Composite sample (Head={head_rows} + Tail={tail_rows} + Random={random_rows}) "
                    f"from {total_in_file} total rows. Sample is SHUFFLED for unbiased distribution."
                )
            else:
                sampling_description = "SAMPLING: Full dataset loaded (no sampling required)."

            metadata_str = f"""
            Rows in Sample: {shape[0]}, Columns: {shape[1]}
            Filesize: {file_size_mb:.2f} MB
            {sampling_description}

            KEY COLUMNS (Top 50 Importance):
            {profile['column_details']}

            REMAINING COLUMNS SUMMARY:
            {profile.get('remaining_columns_summary', 'N/A')}

            AMBIGUITY REPORT:
            {profile['ambiguities']}

            COLUMN GLOSSARY (automated guesses from column names — validate against sample data):
            {profile['glossary']}

            {profile['alerts']}

            Example Rows (from shuffled composite sample):
            {profile['examples']}
            """

            from src.utils.prompting import render_prompt

            SYSTEM_PROMPT_TEMPLATE = """
            You are the Senior Data Steward.

            MISSION: Support the Business Objective: "$business_objective"

            INPUT DATA PROFILE:
            $metadata_str

            IMPORTANT SAMPLING NOTE:
            The profile is based on a COMPOSITE SAMPLE (Head + Tail portions of the data) to capture
            temporal/distributional range. This means you're seeing data from BOTH the beginning AND
            end of the file, which is crucial for time-sorted or grouped datasets.

            YOUR TASK:
            Produce a data summary that gives a downstream
            strategist and planner everything they need to design a modeling approach for
            this business objective. Reason like a senior data scientist seeing this
            dataset for the first time — prioritize what matters most for the stated
            objective, not a generic checklist.

            Focus your analysis on:
            - What business domain does this data represent, and what do the key variables
              mean relative to the objective?
            - Which data quality issues could block or degrade modeling? Prioritize by
              impact, not just by presence (e.g., 5% nulls in a non-critical column is
              less important than mixed types in the likely target).
            - Which columns are identifiers, dates, features, or targets — and why that
              classification matters for this specific objective.
            - Any columns whose meaning is ambiguous or overloaded.
            - If the column glossary contains heuristic hints, validate them against the
              sample data — they are automated guesses, not confirmed roles.

            Be concise and direct. Plain text only.
            """
            
            system_prompt = render_prompt(
                SYSTEM_PROMPT_TEMPLATE,
                business_objective=business_objective,
                metadata_str=metadata_str
            )

            summary, response, finish_reason = self._run_text_prompt(system_prompt)

            # Diagnostic logging for empty responses (best-effort, no PII)
            try:
                text_len = len(summary)
                print(f"STEWARD_LLM_DIAG: text_len={text_len} finish_reason={finish_reason}")
                error_classification = None
                if text_len == 0:
                    print(
                        f"STEWARD_LLM_EMPTY_RESPONSE: finish_reason={finish_reason} "
                        f"prompt_length_chars={len(system_prompt)}"
                    )
                    error_classification = "EMPTY"
                elif text_len < 50:
                    error_classification = "TOO_SHORT"
                trace = {
                    "model": self.model_name,
                    "response_text_len": text_len,
                    "prompt_text_len": len(system_prompt),
                    "finish_reason": str(finish_reason),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "error_classification": error_classification,
                }
                try:
                    os.makedirs("data", exist_ok=True)
                    import json as _json
                    with open("data/steward_llm_trace.json", "w", encoding="utf-8") as f:
                        _json.dump(trace, f, indent=2)
                except Exception:
                    pass
            except Exception as diag_err:
                print(f"STEWARD_LLM_DIAG_WARNING: {diag_err}")
            if not summary or len(summary) < 10:
                # Fallback deterministic summary to avoid blank output
                shape = df.shape
                cols = [str(c) for c in df.columns[:20]]
                null_sample = df.isna().mean().round(3).to_dict()
                summary = (
                    f"DATA SUMMARY: Fallback deterministic summary. Rows={shape[0]}, Cols={shape[1]}, "
                    f"Columns={cols}. Null_frac_sample={null_sample}"
                )
            
            # Add prefix if missing (downstream consumers may display it)
            if not summary.startswith("DATA SUMMARY"):
                summary = "DATA SUMMARY:\n" + summary

            return {
                "summary": summary, 
                "encoding": detected_encoding,
                "sep": sep,
                "decimal": decimal,
                "file_size_bytes": file_size,
                "profile": dataset_profile,
            }
            
        except Exception as e:
            return {
                "summary": f"DATA SUMMARY: Critical Error analyzing data: {e}", 
                "encoding": detected_encoding,
                "sep": sep, 
                "decimal": decimal,
                "profile": {},
            }

    def _clean_json(self, text: str) -> str:
        text = re.sub(r"```json", "", text)
        text = text.replace("```", "")
        return text.strip()

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        cleaned = self._clean_json(text or "")
        if not cleaned:
            return {}
        try:
            parsed = json.loads(cleaned)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            pass
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = cleaned[start : end + 1]
            try:
                parsed = json.loads(snippet)
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}

    def _build_json_repair_prompt(
        self,
        *,
        task_label: str,
        required_keys: List[str],
        raw_output: str,
        parse_error: str,
    ) -> str:
        from src.utils.prompting import render_prompt

        template = """
You are repairing a broken JSON draft for task: $task_label.

Return ONLY raw JSON. No markdown, no prose, no code fences.
Preserve all valid fields from the draft; only fix structure/syntax and ensure required keys exist.

Required keys:
$required_keys

Parse error:
$parse_error

Broken JSON draft:
$raw_output
"""
        return render_prompt(
            template,
            task_label=task_label,
            required_keys=json.dumps(required_keys, ensure_ascii=True),
            parse_error=parse_error,
            raw_output=raw_output or "",
        )

    def _generate_json_payload(
        self,
        *,
        prompt: str,
        task_label: str,
        required_keys: Optional[List[str]] = None,
        max_attempts: int = 3,
    ) -> Dict[str, Any]:
        required_keys = [str(k) for k in (required_keys or []) if k]
        current_prompt = prompt
        last_text = ""
        for attempt in range(max(1, int(max_attempts))):
            text, _response, _finish_reason = self._run_text_prompt(current_prompt)
            last_text = text
            parsed = self._parse_json_response(text)
            has_required = bool(parsed) and all(key in parsed for key in required_keys)
            if parsed and (not required_keys or has_required):
                return parsed
            if attempt >= max(1, int(max_attempts)) - 1:
                break
            missing_keys = [key for key in required_keys if key not in parsed]
            parse_error = "empty_or_invalid_json"
            if missing_keys:
                parse_error = f"missing_required_keys:{missing_keys}"
            current_prompt = self._build_json_repair_prompt(
                task_label=task_label,
                required_keys=required_keys,
                raw_output=text,
                parse_error=parse_error,
            )
        fallback = self._parse_json_response(last_text)
        return fallback if isinstance(fallback, dict) else {}

    def decide_semantics_pass1(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        from src.utils.prompting import render_prompt

        retry_note = ""
        if isinstance(payload, dict) and payload.get("retry_reason"):
            retry_note = f"Retry note: {payload.get('retry_reason')}"

        SYSTEM_PROMPT_TEMPLATE = """
You are the Senior Data Steward.

TASK: Analyze the dataset and business objective to propose semantic hypotheses for modeling. Decide which column is the most likely prediction target, and request the specific evidence you need to confirm your hypotheses.

Business objective:
$business_objective

Column inventory preview (head/tail + count):
$column_inventory_preview

Full column list path: $column_inventory_path

Compact data atlas summary:
$data_atlas_summary

Sample rows (head/tail/random):
$sample_rows

YOUR TASK:
Reason about the dataset and business objective to propose semantic hypotheses.
Think like a senior data scientist deciding how to structure this data for modeling:

- What is the user trying to predict, classify, or optimize?
- Which column best represents that outcome? If multiple candidates exist, pick
  the strongest and explain why in your notes.
- Which columns are row identifiers or potential train/test split markers?
- What evidence do you still need to confirm your hypotheses? Request specific
  measurements (missingness, unique value counts, column profiles) for the
  columns where uncertainty remains.

OUTPUT FORMAT (JSON only, no markdown):
{
  "primary_target": "<your best candidate column name>",
  "split_candidates": ["<columns that may define train/test partitions, or [] if none>"],
  "id_candidates": ["<columns that look like row identifiers, or [] if none>"],
  "evidence_requests": [
    {"kind": "missingness"|"uniques"|"column_profile", "column": "<exact_column_name>", "max_unique": <int, optional for uniques>}
  ],
  "notes": ["Explain your reasoning: why this target, what uncertainties remain, what the evidence will clarify"]
}

CONSTRAINTS:
- evidence_requests: only columns present in inventory/atlas. Request what you
  genuinely need — focus on columns where the answer changes your decision.
- Do NOT invent metrics or statistics you haven't seen. State what you observe and what you need to verify.

$retry_note
"""
        prompt = render_prompt(
            SYSTEM_PROMPT_TEMPLATE,
            business_objective=payload.get("business_objective", ""),
            column_inventory_preview=json.dumps(payload.get("column_inventory_preview", {}), ensure_ascii=True),
            column_inventory_path=payload.get("column_inventory_path", "data/column_inventory.json"),
            data_atlas_summary=str(payload.get("data_atlas_summary", "") or ""),
            sample_rows=json.dumps(payload.get("sample_rows", {}), ensure_ascii=True),
            retry_note=retry_note,
        )
        return self._generate_json_payload(
            prompt=prompt,
            task_label="steward_semantics_pass1",
            required_keys=["primary_target", "split_candidates", "id_candidates", "evidence_requests"],
            max_attempts=3,
        )

    def decide_semantics_pass2(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        from src.utils.prompting import render_prompt

        SYSTEM_PROMPT_TEMPLATE = """
You are the Senior Data Steward.

TASK: Finalize dataset semantics using measured evidence. You already selected the primary target.

Business objective:
$business_objective

Primary target (chosen by you):
$primary_target

Measured target missingness:
$target_missingness

Split candidates (with unique values evidence):
$split_candidates_uniques

Directed evidence bundle (requested in pass1):
$evidence_bundle

Column inventory preview (head/tail + count):
$column_inventory_preview

Full column list path: $column_inventory_path

Compact data atlas summary:
$data_atlas_summary

OUTPUT REQUIREMENTS (JSON ONLY):
{
  "dataset_semantics": {
    "primary_target": "<col>",
    "split_candidates": ["..."],
    "id_candidates": ["..."],
    "target_analysis": {
      "primary_target": "<col>",
      "target_null_frac_exact": <float|null>,
      "target_missing_count_exact": <int|null>,
      "target_total_count_exact": <int|null>,
      "partial_label_detected": <true|false>,
      "labeled_row_heuristic": "target_not_missing",
      "notes": ["..."]
    },
    "partition_analysis": {
      "partition_columns": ["..."],
      "partition_values": {"col": ["v1", "v2"]}
    },
    "notes": ["..."]
  },
  "dataset_training_mask": {
    "training_rows_rule": "...",
    "scoring_rows_rule_primary": "...",
    "scoring_rows_rule_secondary": "...",
    "rationale": ["..."]
  },
  "column_sets": {
    "explicit_columns": ["..."],
    "sets": [
      {"name": "SET_1", "selector": {"type": "prefix_numeric_range", "prefix": "...", "start": 0, "end": 783}},
      {"name": "SET_2", "selector": {"type": "regex", "pattern": "^feature_\\\\d+$"}},
      {"name": "SET_3", "selector": {"type": "all_numeric_except", "except_columns": ["..."]}},
      {"name": "SET_4", "selector": {"type": "all_columns_except", "except_columns": ["..."]}}
    ]
  }
}

YOUR TASK:
Use the measured evidence to finalize dataset semantics. Think like a senior data
scientist validating hypotheses with data — trust measurements over assumptions.

Key reasoning areas:

TARGET VALIDATION:
  You selected primary_target in pass1. Validate this choice against the measured
  evidence. If the evidence confirms it, proceed. If the evidence reveals structural
  problems (e.g., extreme missingness, constant values, or a column that is actually
  a post-decision indicator), document the issue in your notes and flag it — but keep
  primary_target consistent with the value above so downstream agents have a stable
  reference. Use the notes field to communicate any concerns.

TRAINING MASK DESIGN:
  Examine the measured target missingness. What fraction of labels is missing?
  Reason about whether unlabeled rows should be excluded from training, used for
  scoring, or treated differently. Express your reasoning in the rationale field.

PARTITION ANALYSIS:
  Examine the split candidates and their unique values. Do any columns define a
  natural train/test partition? Use the evidence_bundle as your source of truth —
  if it conflicts with sample rows, trust the evidence_bundle.

COLUMN SET DESIGN:
  Design column_sets using compact selectors (prefix_numeric_range, regex,
  all_numeric_except, all_columns_except). Reason with senior-level rigor:
  - Every column should land in a deliberate set. If you use an all_columns_except
    catch-all, review what falls into it — columns with extreme missingness, free-text,
    or columns you flagged as risky should not silently land in a modeling set.
  - Ensure consistency: if you flag a column as a leakage risk or post-decision
    indicator in notes, it must be reflected in a leakage/exclusion set, not left
    in a modeling catch-all.
  - Columns that serve as row-filtering signals (debug flags, synthetic markers)
    should be explicitly classified so downstream agents know their role.

CONSTRAINTS:
- primary_target must match the value provided above (for downstream stability).
  Use notes to flag concerns rather than changing the target unilaterally.
- Output JSON only. No markdown, no extra text.
"""
        prompt = render_prompt(
            SYSTEM_PROMPT_TEMPLATE,
            business_objective=payload.get("business_objective", ""),
            primary_target=payload.get("primary_target", ""),
            target_missingness=json.dumps(payload.get("target_missingness", {}), ensure_ascii=True),
            split_candidates_uniques=json.dumps(payload.get("split_candidates_uniques", []), ensure_ascii=True),
            evidence_bundle=json.dumps(payload.get("evidence_bundle", {}), ensure_ascii=True),
            column_inventory_preview=json.dumps(payload.get("column_inventory_preview", {}), ensure_ascii=True),
            column_inventory_path=payload.get("column_inventory_path", "data/column_inventory.json"),
            data_atlas_summary=str(payload.get("data_atlas_summary", "") or ""),
        )
        return self._generate_json_payload(
            prompt=prompt,
            task_label="steward_semantics_pass2",
            required_keys=["dataset_semantics", "dataset_training_mask", "column_sets"],
            max_attempts=3,
        )

    def _detect_csv_dialect(self, data_path: str, encoding: str) -> Dict[str, str]:
        """
        Robustly detects separator and decimal using csv.Sniffer and internal heuristics.
        """
        try:
            with open(data_path, 'r', encoding=encoding) as f:
                sample = f.read(50000) # 50KB sample
            
            # 1. Delimiter Detection
            try:
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(sample, delimiters=[',', ';', '\t', '|'])
                sep = dialect.delimiter
            except:
                # Fallback Heuristic
                if sample.count(';') > sample.count(','):
                    sep = ';'
                else:
                    sep = ','
            
            # 2. Decimal Detection
            decimal = self._detect_decimal(sample)
            diagnostics: Dict[str, Any] = {}
            if sep == decimal:
                diagnostics["ambiguous_sep_decimal"] = True
                if sep == ",":
                    alt_decimal = "."
                elif sep == ";":
                    alt_decimal = ","
                else:
                    alt_decimal = "," if sep == "." else "."
                diagnostics["decimal_candidates"] = [decimal, alt_decimal]
                diagnostics["selected_decimal"] = alt_decimal
                decimal = alt_decimal

            return {"sep": sep, "decimal": decimal, "diagnostics": diagnostics}
            
        except Exception as e:
            print(f"Steward: Dialect detection failed ({e}). Defaulting to standard.")
            return {"sep": ",", "decimal": "."}

    def _detect_decimal(self, text: str) -> str:
        """
        Analyzes numeric patterns to decide between '.' and ',' as decimal separator.
        """
        # Look for explicit float patterns: 123.45 vs 123,45
        dot_floats = re.findall(r'\d+\.\d+', text)
        comma_floats = re.findall(r'\d+,\d+', text)
        
        # We need to distinguish "comma as thousands sep" from "comma as decimal"
        # Heuristic: If we see many "123,45" but few "123.45", it's likely European.
        # However, "1,000" (thousands) vs "1,000" (small decimal) is hard.
        # Better simple check: 
        # If sep is ';', likely decimal is ','
        # If sep is ',', likely decimal is '.'
        
        # Let's count occurrences
        if len(comma_floats) > len(dot_floats) * 2:
            return ','
        
        return '.'

    def _smart_profile(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Generates intelligent profile: High Card checks, Constant check, Target Detection.
        """
        alerts = ""
        col_details = ""
        ambiguities = ""
        glossary = ""
        remaining_columns_summary = ""

        # Preserve original order; avoid heuristic target suggestions.
        all_cols = df.columns.tolist()
        sorted_cols = all_cols[:50]
        remaining_cols = all_cols[50:]

        def _norm_header(name: str) -> str:
            cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_").lower()
            return re.sub(r"_+", "_", cleaned)

        name_collisions = {}
        for col in all_cols:
            normed = _norm_header(col)
            if normed:
                name_collisions.setdefault(normed, []).append(str(col))

        spaced_cols = [c for c in all_cols if " " in str(c) or "\t" in str(c)]
        trimmed_cols = [c for c in all_cols if str(c) != str(c).strip()]
        punct_cols = [c for c in all_cols if re.search(r"[^0-9A-Za-z_ ]", str(c))]
        collision_examples = [f"{k}: {v}" for k, v in name_collisions.items() if len(v) > 1]

        if trimmed_cols:
            sample = trimmed_cols[:5]
            ambiguities += f"- Column names have leading/trailing whitespace (e.g., {sample}); preserve exact names and account for whitespace when matching.\n"
        if spaced_cols:
            sample = spaced_cols[:5]
            ambiguities += f"- Column names contain spaces (e.g., {sample}); preserve exact names and use explicit mapping if needed.\n"
        if punct_cols:
            sample = punct_cols[:5]
            ambiguities += f"- Column names contain punctuation/special chars (e.g., {sample}); preserve exact names and map carefully.\n"
        if collision_examples:
            sample = collision_examples[:3]
            ambiguities += f"- Canonicalization collisions after normalization (examples: {sample}); disambiguate in mapping.\n"

        for col in sorted_cols:
            dtype = str(df[col].dtype)
            n_unique = df[col].nunique()
            from src.utils.missing import is_effectively_missing_series
            null_pct = is_effectively_missing_series(df[col]).mean()
            
            # Cardinality Check
            unique_ratio = n_unique / len(df) if len(df) > 0 else 0
            
            card_tag = ""
            if unique_ratio > 0.98 and n_unique > 50:
                card_tag = "[HIGH CARDINALITY]"
            elif n_unique <= 1:
                card_tag = "[CONSTANT/USELESS]"
                alerts += f"- ALERT: '{col}' is constant (Value: {df[col].dropna().unique()}).\n"
            
            col_details += f"- {col}: {dtype}, Unique={n_unique} {card_tag}, Nulls={null_pct:.1%}\n"

            if dtype == "object":
                series = df[col].dropna().astype(str)
                if not series.empty:
                    sample = series.sample(min(len(series), 50), random_state=42)
                    percent_like = sample.str.contains("%").mean()
                    comma_decimal = sample.str.contains(r"\d+,\d+").mean()
                    dot_decimal = sample.str.contains(r"\d+\.\d+").mean()
                    numeric_like = sample.str.contains(r"^[\s\-\+]*[\d,.\s%]+$").mean()
                    whitespace = sample.str.contains(r"^\s+|\s+$").mean()
                    if numeric_like > 0.6:
                        ambiguities += f"- {col}: numeric-looking strings (~{numeric_like:.0%}); may need numeric conversion.\n"
                    if percent_like > 0.1:
                        ambiguities += f"- {col}: percent sign present (~{percent_like:.0%}); may need percent normalization.\n"
                    if comma_decimal > 0.1 and dot_decimal < 0.1:
                        ambiguities += f"- {col}: comma decimal pattern (~{comma_decimal:.0%}); likely decimal=','.\n"
                    if whitespace > 0.1:
                        ambiguities += f"- {col}: leading/trailing spaces (~{whitespace:.0%}); strip whitespace.\n"

            tokens = [t for t in col.lower().split("_") if t]
            role_hints = []
            if any(t in {"id", "uuid", "key"} for t in tokens):
                role_hints.append("identifier")
            if any(t in {"date", "fecha", "fec", "time"} for t in tokens):
                role_hints.append("date/time")
            if any(t in {"score", "risk", "rating"} for t in tokens):
                role_hints.append("score")
            if any(t in {"amount", "importe", "price", "cost", "monto"} for t in tokens):
                role_hints.append("monetary")
            if any(t in {"pct", "percent", "ratio", "rate"} for t in tokens):
                role_hints.append("percentage/ratio")
            if any(t in {"flag", "is", "has", "impacto", "status"} for t in tokens):
                role_hints.append("binary/flag")
            if role_hints:
                sample_vals = df[col].dropna().astype(str).head(3).tolist()
                glossary += f"- {col}: dtype={dtype}, heuristic_guess={role_hints} [VALIDATE], sample={sample_vals}\n"

        if remaining_cols:
            type_counts = {"numeric": 0, "categorical": 0, "boolean": 0, "datetime": 0, "other": 0}
            numeric_min = None
            numeric_max = None
            for col in remaining_cols:
                series = df[col]
                dtype = series.dtype
                try:
                    if pd.api.types.is_bool_dtype(dtype):
                        type_counts["boolean"] += 1
                        continue
                    if pd.api.types.is_datetime64_any_dtype(dtype):
                        type_counts["datetime"] += 1
                        continue
                    numeric_probe = pd.to_numeric(series, errors="coerce")
                    numeric_ratio = float(numeric_probe.notna().mean())
                    if numeric_ratio >= 0.8:
                        type_counts["numeric"] += 1
                        if numeric_probe.notna().any():
                            cmin = float(numeric_probe.min())
                            cmax = float(numeric_probe.max())
                            numeric_min = cmin if numeric_min is None else min(numeric_min, cmin)
                            numeric_max = cmax if numeric_max is None else max(numeric_max, cmax)
                    elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
                        type_counts["categorical"] += 1
                    else:
                        type_counts["other"] += 1
                except Exception:
                    type_counts["other"] += 1
            type_chunks = [f"{k}={v}" for k, v in type_counts.items() if int(v) > 0]
            range_text = ""
            if numeric_min is not None and numeric_max is not None:
                range_text = f", numeric_range=[{round(float(numeric_min), 4)}, {round(float(numeric_max), 4)}]"
            remaining_columns_summary = (
                f"Remaining columns ({len(remaining_cols)} beyond top-50): "
                + ", ".join(type_chunks)
                + range_text
            )
            
        # Representative Examples
        try:
            examples = df.sample(min(len(df), 3), random_state=42).to_string(index=False)
        except:
            examples = df.head(3).to_string(index=False)
            
        return {
            "column_details": col_details,
            "alerts": alerts,
            "ambiguities": ambiguities or "None detected.",
            "glossary": glossary or "None.",
            "remaining_columns_summary": remaining_columns_summary,
            "examples": examples
        }


def _infer_type_hint(series: pd.Series) -> str:
    dtype = str(series.dtype)
    if dtype.startswith("int") or dtype.startswith("float"):
        return "numeric"
    if dtype == "bool":
        return "boolean"
    if dtype.startswith("datetime"):
        return "datetime"
    if dtype == "object":
        sample = series.dropna().astype(str).head(200)
        if sample.empty:
            return "categorical"
        try:
            parsed = pd.to_datetime(sample, errors="coerce", dayfirst=True)
            if parsed.notna().mean() > 0.7:
                return "datetime"
        except Exception:
            pass
        numeric_like = sample.str.contains(r"^[\s\-\+]*[\d,.\s%]+$").mean()
        if numeric_like > 0.7:
            return "numeric"
        return "categorical"
    return "unknown"


def detect_pii_findings(df: pd.DataFrame, threshold: float = 0.3) -> Dict[str, Any]:
    patterns = {
        "EMAIL": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", re.IGNORECASE),
        "PHONE": re.compile(r"(?:\+\d{1,3})?[-. (]*\d{3}[-. )]*\d{3}[-. ]*\d{4}", re.IGNORECASE),
        "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
        "IBAN": re.compile(r"[a-zA-Z]{2}\d{2}[a-zA-Z0-9]{4,}", re.IGNORECASE),
    }
    findings: List[Dict[str, Any]] = []
    object_cols = df.select_dtypes(include=["object"]).columns
    for col in object_cols:
        values = df[col].dropna().astype(str).tolist()
        if not values:
            continue
        sample = values[: min(len(values), 200)]
        for pii_type, pattern in patterns.items():
            match_count = sum(1 for val in sample if pattern.search(val))
            ratio = match_count / max(len(sample), 1)
            if ratio >= threshold:
                findings.append(
                    {
                        "column": col,
                        "pii_type": pii_type,
                        "match_ratio": round(ratio, 4),
                    }
                )
                break
    return {"detected": bool(findings), "findings": findings}


def build_dataset_profile(
    df: pd.DataFrame,
    objective: str,
    dialect_info: Dict[str, Any],
    encoding: str,
    file_size_bytes: int,
    was_sampled: bool,
    sample_size: int,
    pii_findings: Dict[str, Any] | None = None,
    sampling_metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    def _safe_numeric_summary(series: pd.Series) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        try:
            numeric = pd.to_numeric(series, errors="coerce")
        except Exception:
            return summary
        count = int(numeric.notna().sum())
        summary["count"] = count
        if count == 0:
            return summary
        try:
            summary.update(
                {
                    "mean": float(numeric.mean()),
                    "std": float(numeric.std()),
                    "min": float(numeric.min()),
                    "q25": float(numeric.quantile(0.25)),
                    "median": float(numeric.median()),
                    "q75": float(numeric.quantile(0.75)),
                    "max": float(numeric.max()),
                }
            )
            summary["zero_frac"] = float((numeric == 0).mean())
            summary["neg_frac"] = float((numeric < 0).mean())
            summary["pos_frac"] = float((numeric > 0).mean())
        except Exception:
            return summary
        return summary

    def _safe_text_summary(series: pd.Series, max_samples: int = 5000) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        try:
            values = series.dropna().astype(str)
        except Exception:
            return summary
        if values.empty:
            summary["count"] = 0
            return summary
        if len(values) > max_samples:
            values = values.sample(max_samples, random_state=42)
        lengths = values.str.len()
        try:
            summary["count"] = int(len(values))
            summary["avg_len"] = float(lengths.mean())
            summary["min_len"] = int(lengths.min())
            summary["max_len"] = int(lengths.max())
            summary["empty_frac"] = float((values.str.strip() == "").mean())
            summary["whitespace_frac"] = float(
                (values.str.len() != values.str.strip().str.len()).mean()
            )
            summary["numeric_like_ratio"] = float(
                values.str.contains(r"^[\s\-\+]*[\d,.\s%]+$", regex=True).mean()
            )
            summary["percent_like_ratio"] = float(values.str.contains(r"%").mean())
            try:
                parsed = pd.to_datetime(values, errors="coerce", dayfirst=True)
                summary["datetime_like_ratio"] = float(parsed.notna().mean())
            except Exception:
                summary["datetime_like_ratio"] = 0.0
        except Exception:
            return summary
        return summary

    columns = [str(c) for c in df.columns]
    type_hints = {col: _infer_type_hint(df[col]) for col in columns}
    missing_frac: Dict[str, float] = {}
    cardinality: Dict[str, Any] = {}
    numeric_summary: Dict[str, Any] = {}
    text_summary: Dict[str, Any] = {}

    from src.utils.missing import is_effectively_missing_series

    for col in columns:
        series = df[col]
        try:
            missing_frac[col] = float(is_effectively_missing_series(series).mean())
        except Exception:
            missing_frac[col] = float(series.isna().mean())
        n_unique = int(series.nunique(dropna=True))
        top_values = []
        try:
            counts = series.astype(str).value_counts(dropna=False).head(5)
            top_values = [{"value": str(idx), "count": int(cnt)} for idx, cnt in counts.items()]
        except Exception:
            top_values = []
        cardinality[col] = {"unique": n_unique, "top_values": top_values}
        hint = type_hints.get(col)
        if hint == "numeric":
            numeric_summary[col] = _safe_numeric_summary(series)
        elif hint in {"categorical", "datetime", "unknown"}:
            text_summary[col] = _safe_text_summary(series)

    duplicate_rows = 0
    duplicate_frac = 0.0
    try:
        duplicate_rows = int(df.duplicated().sum())
        duplicate_frac = float(duplicate_rows / max(len(df), 1))
    except Exception:
        duplicate_rows = 0
        duplicate_frac = 0.0

    rows_for_compute = int(df.shape[0])
    if isinstance(sampling_metadata, dict):
        total_rows = sampling_metadata.get("total_rows_in_file")
        if isinstance(total_rows, (int, float)) and int(total_rows) > 0:
            rows_for_compute = int(total_rows)
    cols_for_compute = int(df.shape[1])
    estimated_memory_mb = round(float(rows_for_compute * cols_for_compute * 8 / 1_000_000), 2)
    if rows_for_compute < 5_000:
        scale_category = "small"
    elif rows_for_compute < 50_000:
        scale_category = "medium"
    elif rows_for_compute < 500_000:
        scale_category = "large"
    else:
        scale_category = "xlarge"
    compute_hints = {
        "estimated_memory_mb": estimated_memory_mb,
        "scale_category": scale_category,
        "cross_validation_feasible": rows_for_compute < 100_000,
        "deep_learning_feasible": rows_for_compute > 10_000 and cols_for_compute < 1_000,
        "rows_for_estimate": rows_for_compute,
        "cols_for_estimate": cols_for_compute,
    }

    profile = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": columns,
        "type_hints": type_hints,
        "missing_frac": missing_frac,
        "cardinality": cardinality,
        "numeric_summary": numeric_summary,
        "text_summary": text_summary,
        "duplicate_stats": {
            "row_dup_count": duplicate_rows,
            "row_dup_frac": round(duplicate_frac, 6),
        },
        "pii_findings": pii_findings or {"detected": False, "findings": []},
        "sampling": {
            "was_sampled": bool(was_sampled),
            "sample_size": int(sample_size),
            "file_size_bytes": int(file_size_bytes),
            # Include detailed sampling metadata for transparency
            "strategy": (sampling_metadata or {}).get("strategy", "unknown"),
            "head_rows": (sampling_metadata or {}).get("head_rows", 0),
            "tail_rows": (sampling_metadata or {}).get("tail_rows", 0),
            "random_rows": (sampling_metadata or {}).get("random_rows", 0),
            "total_rows_in_file": (sampling_metadata or {}).get("total_rows_in_file"),
            "shuffled": (sampling_metadata or {}).get("shuffled", False),
        },
        "dialect": {
            "sep": dialect_info.get("sep"),
            "decimal": dialect_info.get("decimal"),
            "encoding": encoding,
            "diagnostics": dialect_info.get("diagnostics") or {},
        },
        "compute_hints": compute_hints,
    }
    return profile


def write_dataset_profile(profile: Dict[str, Any], path: str = "data/dataset_profile.json") -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        import json as _json
        with open(path, "w", encoding="utf-8") as f:
            _json.dump(profile, f, indent=2, ensure_ascii=True)
    except Exception:
        return


# ============================================================================
# SENIOR REASONING: UNIVERSAL DATA PROFILE (Evidence Layer)
# ============================================================================

# ── Token sets removed (seniority refactoring) ──────────────────────────
# Split and temporal column detection is now handled by the LLM semantic
# passes (decide_semantics_pass1/pass2). The following are kept as minimal
# structural hints only for the evidence-layer profiling, not for
# classification decisions.
# ─────────────────────────────────────────────────────────────────────────


def _infer_temporal_granularity(seconds: float) -> str:
    if seconds <= 0:
        return "unknown"
    if seconds < 90:
        return "sub-minute"
    if seconds < 5400:
        return "hourly"
    if seconds < 172800:
        return "daily"
    if seconds < 1209600:
        return "weekly"
    if seconds < 38016000:
        return "monthly"
    return "yearly"


def _compute_temporal_analysis(
    df: pd.DataFrame,
    columns: List[str],
    max_candidates: int = 30,
    max_rows: int = 10000,
) -> Dict[str, Any]:
    candidates: List[str] = []
    # Use dtype-based detection + minimal structural hints for temporal candidates
    _temporal_hints = {"date", "time", "timestamp", "datetime", "month", "year", "day", "week", "hour"}
    for col in columns:
        tokenized = str(col).lower().replace("-", "_")
        if any(tok in tokenized for tok in _temporal_hints):
            candidates.append(col)
        elif hasattr(df[col], "dtype") and pd.api.types.is_datetime64_any_dtype(df[col]):
            candidates.append(col)
    if not candidates:
        return {"is_time_series": False, "detected_datetime_columns": [], "details": []}

    sample_df = df
    if len(df) > max_rows:
        try:
            sample_df = df.sample(max_rows, random_state=42)
        except Exception:
            sample_df = df.head(max_rows)

    details: List[Dict[str, Any]] = []
    for col in candidates[:max_candidates]:
        try:
            parsed = pd.to_datetime(sample_df[col], errors="coerce", dayfirst=True)
            parse_ratio = float(parsed.notna().mean())
            if parse_ratio < 0.6:
                continue
            non_null = parsed.dropna()
            is_sorted = bool(non_null.is_monotonic_increasing) if len(non_null) > 1 else False
            median_seconds = None
            granularity = "unknown"
            if len(non_null) > 2:
                diffs = non_null.sort_values().diff().dropna().dt.total_seconds()
                if not diffs.empty:
                    median_seconds = float(diffs.median())
                    granularity = _infer_temporal_granularity(median_seconds)
            details.append(
                {
                    "column": col,
                    "parse_ratio": round(parse_ratio, 4),
                    "is_sorted_ascending": is_sorted,
                    "median_step_seconds": round(median_seconds, 3) if isinstance(median_seconds, (int, float)) else None,
                    "granularity_hint": granularity,
                }
            )
        except Exception:
            continue

    is_time_series = any(bool(item.get("is_sorted_ascending")) for item in details)
    return {
        "is_time_series": bool(is_time_series),
        "detected_datetime_columns": [str(item.get("column")) for item in details if item.get("column")],
        "details": details[:20],
    }


def _compute_feature_target_associations(
    df: pd.DataFrame,
    *,
    target_col: str,
    inferred_type: str,
    top_k: int = 20,
    max_numeric_cols: int = 60,
    max_categorical_cols: int = 40,
    max_rows: int = 50000,
) -> List[Dict[str, Any]]:
    if target_col not in df.columns:
        return []
    work = df
    if len(work) > max_rows:
        try:
            work = work.sample(max_rows, random_state=42)
        except Exception:
            work = work.head(max_rows)

    associations: List[Dict[str, Any]] = []
    numeric_candidates: List[str] = []
    categorical_candidates: List[str] = []
    for col in work.columns:
        if col == target_col:
            continue
        series = work[col]
        try:
            numeric_ratio = float(pd.to_numeric(series, errors="coerce").notna().mean())
        except Exception:
            numeric_ratio = 0.0
        if numeric_ratio >= 0.8:
            numeric_candidates.append(col)
        else:
            categorical_candidates.append(col)

    numeric_candidates = numeric_candidates[:max_numeric_cols]
    categorical_candidates = categorical_candidates[:max_categorical_cols]

    if inferred_type == "regression":
        target_num = pd.to_numeric(work[target_col], errors="coerce")
        candidate_scores: List[Tuple[str, float]] = []
        for col in numeric_candidates:
            try:
                feat = pd.to_numeric(work[col], errors="coerce")
                pair = pd.DataFrame({"x": feat, "y": target_num}).dropna()
                if len(pair) < 30:
                    continue
                corr = float(pair["x"].corr(pair["y"]))
                if pd.isna(corr):
                    continue
                candidate_scores.append((col, corr))
            except Exception:
                continue
        candidate_scores.sort(key=lambda item: abs(item[1]), reverse=True)
        for col, corr in candidate_scores[:top_k]:
            associations.append(
                {
                    "column": col,
                    "target": target_col,
                    "method": "pearson",
                    "score": round(abs(float(corr)), 6),
                    "direction": "positive" if corr >= 0 else "negative",
                }
            )
    else:
        # Classification/other: numeric eta^2 + categorical Cramer's V.
        target = work[target_col].astype(str)
        target_non_null = target[target.notna()]
        if target_non_null.empty:
            return []

        # Numeric -> eta squared
        for col in numeric_candidates:
            try:
                feat = pd.to_numeric(work[col], errors="coerce")
                pair = pd.DataFrame({"x": feat, "y": target}).dropna()
                if len(pair) < 50:
                    continue
                grand_mean = float(pair["x"].mean())
                grouped = pair.groupby("y")["x"]
                ss_between = 0.0
                for _, values in grouped:
                    n_g = float(len(values))
                    mean_g = float(values.mean())
                    ss_between += n_g * ((mean_g - grand_mean) ** 2)
                ss_total = float(((pair["x"] - grand_mean) ** 2).sum())
                if ss_total <= 0:
                    continue
                eta_sq = max(0.0, min(1.0, ss_between / ss_total))
                associations.append(
                    {
                        "column": col,
                        "target": target_col,
                        "method": "anova_eta_squared",
                        "score": round(float(eta_sq), 6),
                        "direction": "non_directional",
                    }
                )
            except Exception:
                continue

        # Categorical -> chi2 / Cramer's V
        for col in categorical_candidates:
            try:
                contingency = pd.crosstab(work[col].astype(str), target)
                if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                    continue
                n = float(contingency.values.sum())
                if n <= 0:
                    continue
                try:
                    from scipy.stats import chi2_contingency  # type: ignore

                    chi2, _, _, _ = chi2_contingency(contingency, correction=False)
                except Exception:
                    observed = contingency.values.astype(float)
                    expected = (
                        observed.sum(axis=1, keepdims=True)
                        * observed.sum(axis=0, keepdims=True)
                        / max(observed.sum(), 1.0)
                    )
                    with np.errstate(divide="ignore", invalid="ignore"):
                        chi2_mat = np.where(expected > 0, ((observed - expected) ** 2) / expected, 0.0)
                    chi2 = float(np.nansum(chi2_mat))
                r, k = contingency.shape
                denom = max(min(k - 1, r - 1), 1)
                cramers_v = float(np.sqrt(max(chi2 / n / denom, 0.0)))
                associations.append(
                    {
                        "column": col,
                        "target": target_col,
                        "method": "chi2_cramers_v",
                        "score": round(float(cramers_v), 6),
                        "direction": "non_directional",
                    }
                )
            except Exception:
                continue

        associations.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
        associations = associations[:top_k]

    associations.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    return associations[:top_k]


def _compute_multicollinearity_pairs(
    df: pd.DataFrame,
    *,
    threshold: float = 0.95,
    max_numeric_cols: int = 60,
    max_pairs: int = 200,
) -> List[Dict[str, Any]]:
    numeric_cols: List[str] = []
    for col in df.columns:
        try:
            ratio = float(pd.to_numeric(df[col], errors="coerce").notna().mean())
        except Exception:
            ratio = 0.0
        if ratio >= 0.85:
            numeric_cols.append(str(col))
    if len(numeric_cols) < 2:
        return []
    numeric_cols = numeric_cols[:max_numeric_cols]

    try:
        numeric_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        corr = numeric_df.corr().abs()
    except Exception:
        return []

    pairs: List[Dict[str, Any]] = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            value = corr.iloc[i, j]
            if pd.isna(value):
                continue
            if float(value) >= threshold:
                pairs.append(
                    {
                        "col_a": cols[i],
                        "col_b": cols[j],
                        "corr_abs": round(float(value), 6),
                    }
                )
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def build_data_profile(
    df: pd.DataFrame,
    contract: Dict[str, Any] | None = None,
    analysis_type: str | None = None,
) -> Dict[str, Any]:
    """
    Build a universal data_profile.json for senior reasoning.

    This is the EVIDENCE LAYER - objective facts about the data.
    It does NOT make decisions; it provides evidence for ml_plan.json.

    Args:
        df: The cleaned DataFrame to profile
        contract: Optional execution contract (for outcome_columns, column_roles)
        analysis_type: Optional analysis type (classification, regression, etc.)

    Returns:
        Dictionary with universal profile including:
        - basic_stats: rows, cols, dtypes
        - missingness: per-column missing fraction (top 30)
        - outcome_analysis: if contract has outcome_columns
        - split_candidates: columns whose names suggest split/fold usage
        - constant_columns: columns with <= 1 unique value
        - high_cardinality_columns: columns with unique ratio > 0.95
        - leakage_flags: potential leakage indicators
    """
    contract = contract or {}
    columns = [str(c) for c in df.columns]
    n_rows = int(df.shape[0])
    n_cols = int(df.shape[1])

    # 1. Basic stats: dtypes
    dtypes_map = {}
    for col in columns:
        dtypes_map[col] = str(df[col].dtype)

    # 2. Missingness (top 30 by missing fraction)
    from src.utils.missing import is_effectively_missing_series
    missingness = {}
    for col in columns:
        try:
            miss_frac = float(is_effectively_missing_series(df[col]).mean())
        except Exception:
            miss_frac = float(df[col].isna().mean())
        missingness[col] = round(miss_frac, 4)
    # Sort by missingness descending, take top 30
    sorted_miss = sorted(missingness.items(), key=lambda x: x[1], reverse=True)
    missingness_top30 = dict(sorted_miss[:30])

    # 3. Outcome analysis (if contract specifies outcome_columns)
    outcome_analysis = {}
    outcome_cols = []
    if contract.get("outcome_columns"):
        raw_outcomes = contract.get("outcome_columns")
        if isinstance(raw_outcomes, list):
            outcome_cols = [str(c) for c in raw_outcomes if c and str(c).lower() != "unknown"]
        elif isinstance(raw_outcomes, str) and raw_outcomes.lower() != "unknown":
            outcome_cols = [raw_outcomes]
    # Fallback: column_roles["outcome"]
    if not outcome_cols:
        roles = contract.get("column_roles", {})
        if isinstance(roles, dict):
            outcome_from_roles = roles.get("outcome", [])
            if isinstance(outcome_from_roles, list):
                outcome_cols = [str(c) for c in outcome_from_roles if c]
            elif isinstance(outcome_from_roles, str):
                outcome_cols = [outcome_from_roles]

    for outcome_col in outcome_cols:
        if outcome_col not in df.columns:
            outcome_analysis[outcome_col] = {"present": False, "error": "column_not_found"}
            continue
        series = df[outcome_col]
        non_null_count = int(series.notna().sum())
        total_count = int(len(series))
        null_frac = round(1.0 - (non_null_count / total_count) if total_count > 0 else 0, 4)
        analysis_entry = {
            "present": True,
            "non_null_count": non_null_count,
            "total_count": total_count,
            "null_frac": null_frac,
        }
        # Determine if classification or regression
        n_unique = int(series.nunique(dropna=True))
        inferred_type = analysis_type or ""
        if not inferred_type:
            # Heuristic: if <= 20 unique values, likely classification
            if n_unique <= 20:
                inferred_type = "classification"
            else:
                inferred_type = "regression"
        analysis_entry["inferred_type"] = inferred_type
        analysis_entry["n_unique"] = n_unique

        if inferred_type == "classification":
            # Class counts (capped at 30 classes)
            try:
                counts = series.dropna().value_counts().head(30)
                class_counts = {str(k): int(v) for k, v in counts.items()}
                analysis_entry["n_classes"] = n_unique
                analysis_entry["class_counts"] = class_counts
                if class_counts:
                    min_class = min(class_counts.values())
                    max_class = max(class_counts.values())
                    total_classes = sum(class_counts.values())
                    analysis_entry["class_imbalance_ratio"] = round(
                        float(min_class / max(max_class, 1)), 6
                    )
                    analysis_entry["minority_class_share"] = round(
                        float(min_class / max(total_classes, 1)), 6
                    )
            except Exception:
                pass
        else:
            # Regression: quantiles
            try:
                numeric_series = pd.to_numeric(series, errors="coerce").dropna()
                if len(numeric_series) > 0:
                    quantiles = {
                        "min": float(numeric_series.min()),
                        "q25": float(numeric_series.quantile(0.25)),
                        "median": float(numeric_series.median()),
                        "q75": float(numeric_series.quantile(0.75)),
                        "max": float(numeric_series.max()),
                    }
                    analysis_entry["quantiles"] = quantiles
                    try:
                        analysis_entry["skewness"] = float(numeric_series.skew())
                    except Exception:
                        analysis_entry["skewness"] = None
                    try:
                        analysis_entry["kurtosis"] = float(numeric_series.kurtosis())
                    except Exception:
                        analysis_entry["kurtosis"] = None
                    q1 = float(numeric_series.quantile(0.25))
                    q3 = float(numeric_series.quantile(0.75))
                    iqr = q3 - q1
                    if iqr > 0:
                        low = q1 - 1.5 * iqr
                        high = q3 + 1.5 * iqr
                        outlier_frac = float(((numeric_series < low) | (numeric_series > high)).mean())
                    else:
                        outlier_frac = 0.0
                    analysis_entry["outlier_frac_iqr"] = round(outlier_frac, 6)
            except Exception:
                pass
        outcome_analysis[outcome_col] = analysis_entry

    # 4. Split candidates: columns with structural hints for train/test splitting
    _split_hints = {"split", "set", "fold", "train", "test", "partition", "is_train", "is_test"}
    split_candidates = []
    for col in columns:
        col_lower = col.lower().replace("_", " ").replace("-", " ")
        tokens = set(col_lower.split())
        if tokens & _split_hints:
            # Gather unique values evidence
            try:
                uniques = df[col].dropna().unique()[:20].tolist()
                uniques_str = [str(v) for v in uniques]
            except Exception:
                uniques_str = []
            split_candidates.append({
                "column": col,
                "unique_values_sample": uniques_str,
            })

    # 5. Constant columns (unique values <= 1)
    constant_columns = []
    for col in columns:
        n_uniq = df[col].nunique(dropna=True)
        if n_uniq <= 1:
            constant_columns.append(col)

    # 6. High cardinality columns (unique ratio > 0.95 and > 50 uniques)
    high_cardinality_columns = []
    for col in columns:
        n_uniq = df[col].nunique(dropna=True)
        unique_ratio = n_uniq / n_rows if n_rows > 0 else 0
        if unique_ratio > 0.95 and n_uniq > 50:
            high_cardinality_columns.append({
                "column": col,
                "n_unique": n_uniq,
                "unique_ratio": round(unique_ratio, 4),
            })

    # 7. Leakage flags: if outcome column name appears elsewhere as feature
    leakage_flags = []
    outcome_names_lower = {c.lower() for c in outcome_cols}
    for col in columns:
        col_lower = col.lower()
        # Check if outcome name is substring of another column (potential derived leakage)
        for outcome in outcome_names_lower:
            if outcome in col_lower and col not in outcome_cols:
                leakage_flags.append({
                    "column": col,
                    "reason": f"name_contains_outcome:{outcome}",
                    "severity": "SOFT",
                })

    # 8. Temporal analysis
    temporal_analysis = _compute_temporal_analysis(df, columns)

    # 9. Feature-target associations (top signal features)
    primary_target = ""
    primary_target_type = ""
    for col in outcome_cols:
        entry = outcome_analysis.get(col)
        if isinstance(entry, dict) and entry.get("present"):
            primary_target = col
            primary_target_type = str(entry.get("inferred_type") or "")
            break
    feature_target_associations: List[Dict[str, Any]] = []
    if primary_target:
        try:
            feature_target_associations = _compute_feature_target_associations(
                df,
                target_col=primary_target,
                inferred_type=primary_target_type or "classification",
                top_k=20,
            )
        except Exception:
            feature_target_associations = []

    # 10. Multicollinearity
    multicollinearity_pairs = _compute_multicollinearity_pairs(df, threshold=0.95)

    # 11. Compute hints
    estimated_memory_mb = round(float(n_rows * n_cols * 8 / 1_000_000), 2)
    if n_rows < 5_000:
        scale_category = "small"
    elif n_rows < 50_000:
        scale_category = "medium"
    elif n_rows < 500_000:
        scale_category = "large"
    else:
        scale_category = "xlarge"
    compute_hints = {
        "estimated_memory_mb": estimated_memory_mb,
        "scale_category": scale_category,
        "cross_validation_feasible": n_rows < 100_000,
        "deep_learning_feasible": n_rows > 10_000 and n_cols < 1_000,
    }

    profile = {
        "basic_stats": {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "columns": columns,
        },
        "dtypes": dtypes_map,
        "missingness_top30": missingness_top30,
        "outcome_analysis": outcome_analysis,
        "split_candidates": split_candidates,
        "constant_columns": constant_columns,
        "high_cardinality_columns": high_cardinality_columns,
        "leakage_flags": leakage_flags,
        "temporal_analysis": temporal_analysis,
        "feature_target_associations": feature_target_associations,
        "multicollinearity_pairs_high": multicollinearity_pairs,
        "compute_hints": compute_hints,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return profile


def write_data_profile(profile: Dict[str, Any], path: str = "work/artifacts/data_profile.json") -> None:
    """Write data_profile.json to the specified path."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2, ensure_ascii=True)
    except Exception as e:
        print(f"Warning: failed to write data_profile.json: {e}")
