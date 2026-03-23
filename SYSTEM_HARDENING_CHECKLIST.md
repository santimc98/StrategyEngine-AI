# System Hardening Checklist

## Goal
- Reach high operational confidence that the system preserves good agent reasoning.
- Make every new architecture bug close a failure class, not just one run.

## Non-Negotiable Rule
For every new system bug, all four must happen:
1. Fix the bug.
2. Add a replay using the real run.
3. Add a layer-level invariant/property test.
4. Classify the root cause in the replay corpus inventory.

The bug is not closed unless all four are done.

## Readiness Gates

### Gate 1: Execution Planner Reliability
- `semantic_core` transport is robust for valid/recoverable JSON.
- `contract_compile` does not false-reject semantically valid contracts.
- `semantic_guard` validates typed semantic equivalence, not literal shape only.
- `contract_raw` remains immutable after LLM generation.

Exit criteria:
- No known planner replay run fails for a system-side reason.

### Gate 2: Projection And Runtime Reliability
- Views are lossless subsets of the top-level contract plus additive hints only.
- Produced artifact index matches promoted artifacts.
- Output contract checks use the same artifact truth as the runtime workspace.
- Translator/report manifests do not duplicate or corrupt rich deliverables.

Exit criteria:
- No known runtime replay run fails because of projection/index/publication mismatch.

### Gate 3: Agentic Quality
- Remaining failures are agentic quality problems, not architecture breakage.
- Data engineer, reviewer, strategist, and translator are judged on reasoning quality, not on broken system plumbing.

Exit criteria:
- New failures are predominantly `A1/A2`, not `S1/S2`.

### Gate 4: Commercial Confidence
- Replay corpus passes.
- Property/invariant suite passes.
- Fresh-run streak shows no architecture regressions.

Exit criteria:
- 10-20 new runs in a row with no architecture failure.

## Failure Taxonomy
- `S1`: critical system architecture bug; rejects good contract or breaks a good run.
- `S2`: secondary system bug; does not crash the run but corrupts views, manifests, evidence, or outputs.
- `A1`: critical agentic reasoning bug.
- `A2`: agentic quality bug; run finishes but work is not senior quality.
- `AMB`: legitimate ambiguity; should be adjudicated or downgraded, not hard-failed.

## Daily Operating Loop
1. Classify every new failure as `S1`, `S2`, `A1`, `A2`, or `AMB`.
2. If `S1` or `S2`, add or update replay corpus coverage immediately.
3. Add one invariant/property test for the affected layer.
4. Re-run the replay subset for that layer before touching prompts again.
5. Only work on agent-quality hardening once the relevant architecture layer is green.

## Required Invariants

### Contract And Planner
- The agent-authored contract is never semantically rewritten by deterministic code.
- `required_outputs` semantic equivalence is typed:
  - physical path
  - semantic deliverable
  - filename-like semantic deliverable
- Recoverable JSON truncation at EOF is repaired structurally, not semantically.

### Views
- Top-level contract is the sole semantic source of truth.
- `agent_interfaces` may enrich views but may never narrow canonical content.
- Every active agent view covers all canonical outputs and gates relevant to that agent.

### Runtime And Artifacts
- All owned required outputs promoted from sandbox must exist in workspace and index.
- `produced_artifact_index` and output contract must agree on promoted artifacts.
- Report manifests must extract only real paths from rich deliverables.

## Fresh-Run Success Criteria
A run only counts toward the final confidence streak if all are true:
- no execution planner crash
- no false contract rejection
- no loss of top-level contract coverage in views
- no artifact promotion/index mismatch
- no manifest corruption

## Current Priority Order
1. Close all `execution_planner` architecture classes in the replay corpus.
2. Close artifact publication/index/manifest architecture classes.
3. Then harden agentic execution quality.

## Definition Of "System Supports Agent Reasoning"
The system supports agent reasoning when:
- good contracts survive transport, validation, projection, and runtime unchanged in meaning
- architecture failures stop appearing in fresh runs
- replay corpus and invariants catch regressions before a new run does
