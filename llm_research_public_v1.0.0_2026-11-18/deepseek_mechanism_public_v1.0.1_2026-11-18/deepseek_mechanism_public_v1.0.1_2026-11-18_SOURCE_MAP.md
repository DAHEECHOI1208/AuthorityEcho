# deepseek_mechanism_public_v1.0.1_2026-11-18: source coverage

Version 1.0.1; 2026-11-18. This table maps 49 retained source entries to consolidated executable functions. It is not a claim that every historical CLI, hard-coded data package, or prior numeric output was copied unchanged. Native transport and selected numerical functions are direct ports; orchestration and schema adapters are new.

Original private filenames, research documents, corpus texts, responses, and original absolute paths are omitted. Source byte hashes permit private provenance matching. Old one-off waivers and environment repair scripts are replaced by current-run checks; they do not confer validation on new data.

| Source | Function | Integrated implementation | Scope |
|---|---|---|---|
| S01 | Pinned r12 target, final-answer transport, native extraction | `native/*; runtime.py` | Direct native ports; GPU smoke required |
| S02 | Reference numerical warning policy | `diagnostics.py` | Diagnostic only; no threshold relaxation |
| S03 | Paired harmful/benign groups and H/R validation | `judging.py; representation.py; validation_base.py` | Private paired corpus and preserved split metadata required |
| S04 | Missing secondary judgment handling | `judging.py` | Missing judgments are unknown, never refusal |
| S05 | Conditional H_A and R_H selection | `representation.py` | Original fixed score and tie rule; exploratory |
| S06 | Condition/behavior/representation linkage | `representation.py` | Saved-score paired linkage |
| S07 | C0 baseline score changes | `readouts.py; representation.py` | Common cohorts and separate behavioral denominators |
| S08 | Full corpus with fixed centers | `scoring.py; readouts.py` | No center refit unless explicitly requested |
| S09 | Full corpus center refit and reselection | `joint_bootstrap.py; representation.py` | Original joint fit/evaluate draws; apparent performance only |
| S10 | Refitted center linkage | `representation.py; readouts.py` | Pass explicitly refitted score ledger |
| S11 | Fit-only HA center | `scoring.py` | Equal-parent/equal-condition raw centers |
| S12 | Full-corpus HA center | `scoring.py` | Explicit full500 scope; not independent validation |
| S13 | C0/C1 readouts under two center banks | `scoring.py; readouts.py` | Run each frozen bank separately |
| S14 | HA score C0 contrasts | `representation.py` | a=cos(HA)-cos(BA) |
| S15 | C0 controls, additivity and benign reference | `readouts.py` | Same-parent common cohorts; no causal mediation claim |
| S16 | All-layer g | `scoring.py; readouts.py` | 27 actual block outputs |
| S17 | All-position g | `scoring.py; readouts.py` | Actual t_U/t_D/t_C/t_A; no invented missing positions |
| S18 | Adjusted common/benign regression | `regression.py; regression_base.py` | Original estimators, fold-only scaling, component holdouts and simultaneous bands |
| S19 | C0 one-component restoration | `geometry.py; hooks.py; interventions.py` | Separate 16/20 t_A; optional explicitly evidenced historical cohort |
| S20 | Fresh C0 audit and fresh RA | `interventions.py; analysis.py` | Baseline-only immutable membership |
| S21 | Restore-to-fresh-analysis orchestration | `cli.py` | One entry point; stage journals and resume |
| S22 | Multi-set fresh RA and dual confidence intervals | `inputs.py; analysis.py` | Set IDs preserved; sets not pooled; pointwise and corrected intervals separate |
| S23 | Replaceable question input with resume integrity | `inputs.py; io.py` | No whitelist of old input file hashes; hash frozen at prepare |
| S24 | Strength and common HR reference | `geometry.py; analysis.py; judging.py` | 16 layer alpha 1.5 primary, 2 secondary; contemporary rejudging batch |
| S25 | Whole-vector C0 replacement | `geometry.py; hooks.py; interventions.py` | Own C0 donor; actual BF16 equality; no projection or normalization |
| S26 | C3/C4 probes and surface controls | `probes.py` | Fit-only direct/shared/orthogonal probes; frozen surface controls |
| S27 | Entity/purpose A/B cross inputs | `inputs.py; readouts.py` | Explicit B personas required; no persona fabrication |
| S28 | Additional cross contrasts | `readouts.py` | Aligned/misaligned and difference-in-differences |
| S29 | Cross representation contrasts | `readouts.py` | Original cluster weighting for cross comparisons |
| S30 | Cross behavioral transitions | `readouts.py; analysis.py` | Explicit R/A only; same parents across A/B |
| S31 | Purpose layer12 restoration | `interventions.py; alignment_statistics.py; alignment_report.py` | Original shared direction, A/B donors, follow-up layers12-16 and original report functions |
| S32 | Matched external benign question inputs | `inputs.py` | Private pre-matched corpus accepted; benchmark text not bundled or regenerated |
| S33 | Human audit import/report | `operations.py` | No human label automatically treated as ground truth; no historical overwrite |
| S34 | Historical probe exception | `gates.py` | Old one-off waiver deliberately not transferred; new run must pass its own mechanical gates |
| S35 | Repeat diagnostic | `diagnostics.py` | Optional GPU same-input repeat; no universal determinism claim |
| S36 | Layer-zero trace | `diagnostics.py` | Optional module-level t_U trace |
| S37 | Length sensitivity | `diagnostics.py` | Optional same-input o_proj length variation |
| S38 | Isolated o_proj diagnostic | `diagnostics.py` | Prefix values held fixed; changing matrix shape recorded |
| S39 | All-layer activation export | `runtime.py; scoring.py` | FP32 NPZ and identity-bound ledger |
| S40 | Authentication failure recovery | `operations.py; judge_transport.py` | Archive failed journals; explicit bounded retry; stop on error |
| S41 | Center condition-family sensitivity | `workflows.py` | Four/five/six conditions, fixed layer evaluation |
| S42 | New-set center sensitivity | `workflows.py` | Separate set inputs; no cross-set pooling |
| S43 | Benign alert audit | `readouts.py` | Benign alerts reported separately from refusal |
| S44 | Delta alert audit | `readouts.py` | Select-frozen delta-score threshold; check evaluation |
| S45 | C4 score alert | `readouts.py` | Select-frozen C4 threshold and conservative ties |
| S46 | g readout audit | `readouts.py; representation.py` | Saved scores, class support and explicit unknowns |
| S47 | Follow-up g and additional layer capture | `hooks.py; scoring.py` | All 27 layers retained, including layer15 |
| S48 | Integrated alert/denominator audit | `operations.py; readouts.py` | Separate denominators and missingness; no efficacy auto-pass |
| S49 | Raw NPZ provenance audit | `operations.py; imports.py` | Hash, axes, dtype, finite/missing vectors and identity-indexed legacy cache |

## Added extension

Entity layer12 restoration is implemented alongside purpose layer12 restoration. Both use the same frozen shared congruence direction. Entity is a new implementation, not a claim that a historical entity intervention was completed. A uses AB -> AA; B uses BA -> BB. The input remains the recipient aligned condition.

## Deliberate exclusions

The C3-E-new condition expansion and the 60-question BR supplement are not included. No benchmark question generation, historical corpus redistribution, blanket probe waiver, or secret credential is included. No GPU or provider API has been used to validate this public package.

## Reproduction boundary

This build supports reproducing computation with supplied private inputs and artifacts. It cannot recreate unprovided prompts, reference documents, labels, centers, cluster assignments, or a past mutable Judge backend. New single-case inputs are new observations, not replications of old numerical results. Source-derived calculations with different explicit cohort policies remain separately labeled.

Change history: 1.0.0 consolidates transport, captures, restoration families, saved-result analyses, diagnostics and recovery into one public executable project.

1.0.1 (2026-11-18): benign public example and explicit benign metadata wrapper; research functions and selection rules retained.
