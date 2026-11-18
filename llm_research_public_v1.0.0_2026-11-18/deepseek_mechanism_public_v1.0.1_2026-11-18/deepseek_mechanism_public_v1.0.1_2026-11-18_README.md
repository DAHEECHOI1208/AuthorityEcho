# DeepSeek mechanism public pipeline

Version 1.0.1 — 2026-11-18

This is an executable integrated research package. It includes native model loading, prompt serialization, generation, activation extraction, intervention hooks, Judge transport, frozen-cohort analysis, representation analyses, optional diagnostics, and recovery. `run.py` is the common entry point; the source correspondence table covers 49 retained source entries and a new entity layer12 extension.

The bundled `input.json` is a benign, fictional book-club invitation case, byte-identical to the input in the evaluation package in this bundle. It replaces the previous example. `examples/benign_corpus.json` wraps the same case in explicit benign metadata for the existing DeepSeek loader. The six conditions are C0, C1, C2, C3-E, C3-P and C4. There is no C3-E-new condition. C2 uses the same document as C4 without the persona.

The DeepSeek runtime uses its original post-retrieval final-answer system prompt and explicit tool-history adapter. It does not use the commercial API package's system prompt or flatten its messages into a generic chat request.

## 1. What can run from one input

| Requested operation | Additional requirement | Result with one question |
|---|---|---|
| Six-condition generation | Pinned model/tokenizer and CUDA BF16 environment | Six new responses |
| Activation extraction | Same model/environment | 27 blocks × four positions, FP32 NPZ; t_D absent for C0/C1 |
| Luna judgment | Original private rubric and environment key | New labels; no old labels copied |
| f/g measurements | Frozen private fit250 centers | Scores for supplied center positions; unavailable positions remain null |
| 16/20 component restoration | Centers plus same-run fresh C0 and C4 | Runs only on the pre-intervention eligible cohort |
| Strength and full-vector experiments | Same assets and fixed cohort | Independent new responses, explicit controls |
| Entity/purpose layer12 restoration | Original frozen shared congruence directions | A-persona comparisons; B requires explicit cross templates |
| New center fitting, selection or statistical validation | Private multi-parent data, labels and fit/select/check component assignments | Not estimable from one question |

A single occupied shared component cannot yield a cluster confidence interval. Empty fresh RA produces no intervention tasks and no zero-valued efficacy estimate. Prior numerical thresholds are not redefined to force a pass.

No original harmful corpus, model responses, original research documents, trained centers, private NPZ vectors, personal names, server addresses, SSH keys or API keys are distributed. The original Judge rubric contains non-English text and is loaded from a private external artifact to preserve the historical request exactly. It has not been translated or encoded into this public source. Runtime private data and generated results may contain their original language; the distributed source, documentation and bundled example are English-only. Required third-party license notices are retained.

## 2. Public code versus private working data

Keep all private inputs, rubric exports, centers, runs and backups outside the package directory. Never publish a generated run: it contains question text, answers, Judge payloads and activation vectors.

The main APIs and numerical definitions are preserved. Run IDs, task IDs, directory layout and the unified CLI are new. Native source equivalence, pure numerical comparisons and CPU/mock validation are documented under `verification/`. Real tokenizer/model execution, CUDA kernels and the provider endpoint still require the supplied RunPod smoke commands.

The reference GPU stack is an archived execution contract, not a recommendation to upgrade an existing environment:

- Python 3.12.14; torch 2.14.1+cu130; Transformers 4.57.6.
- DeepSeek-V2-Lite-Chat model and tokenizer revision `85864749cd611b4353ce1decdb286193298f64c7`.
- BF16, no quantization, SDPA, built-in Transformers implementation, no remote model code.
- Original absolute YaRN scale correction, source-hash and configuration guards.
- Greedy generation, seed 42, maximum 4096 new tokens, maximum 2048 input tokens; no input truncation.
- Full versions are in `requirements-gpu.txt` and `configs/environment.json`.

Reuse the matching existing environment. `preflight --gpu` fails on a mismatch; changing the pinned configuration is a new experimental protocol, not a repair to bypass a check.

## 3. Upload and connect — Mac local terminal

Download the ZIP and set the connection values from the RunPod Connect screen. The values below are placeholders, not a saved machine identity.

```bash
export PACKAGE=deepseek_mechanism_public_v1.0.1_2026-11-18
export RUNPOD_HOST='YOUR_RUNPOD_SSH_HOST'
export RUNPOD_PORT='YOUR_RUNPOD_SSH_PORT'
export RUNPOD_SSH_KEY="$HOME/.ssh/id_ed25519"
ssh -i "$RUNPOD_SSH_KEY" -p "$RUNPOD_PORT" "root@$RUNPOD_HOST" 'mkdir -p /workspace/experiment/packages'
scp -i "$RUNPOD_SSH_KEY" -P "$RUNPOD_PORT" "$HOME/Downloads/$PACKAGE.zip" "root@$RUNPOD_HOST:/workspace/experiment/packages/"
ssh -i "$RUNPOD_SSH_KEY" -p "$RUNPOD_PORT" "root@$RUNPOD_HOST"
```

## 4. Unpack and validate — RunPod terminal

```bash
export PACKAGE=deepseek_mechanism_public_v1.0.1_2026-11-18
cd /workspace/experiment/packages
unzip "$PACKAGE.zip"
export CODE="/workspace/experiment/packages/$PACKAGE"
export MECHANISM_PYTHON=/workspace/experiment/.venv/bin/python
cd "$CODE"
sha256sum -c SHA256SUMS
"$MECHANISM_PYTHON" -B -m unittest discover -s tests -v
bash run_stage.sh preflight
```

Set `MECHANISM_PYTHON` to the existing matching Python executable on your server; the example path is not assumed to exist. CPU tests require only `requirements-cpu.txt`. They do not download a model or contact a provider.

For a new CPU-only environment on Mac:

```bash
python3 -m venv "$HOME/.venvs/deepseek-mechanism-cpu"
"$HOME/.venvs/deepseek-mechanism-cpu/bin/python" -m pip install -r requirements-cpu.txt
"$HOME/.venvs/deepseek-mechanism-cpu/bin/python" -B -m unittest discover -s tests -v
```

Do not use this CPU environment for GPU generation. For a new GPU environment, install the exact torch CUDA build and the exact versions in `requirements-gpu.txt`, then run preflight; availability of those archived wheels was not tested here.

If the pinned checkpoint is not already cached, explicitly download it into a persistent volume using the existing environment:

```bash
export HF_HOME=/workspace/experiment/model-cache
"$MECHANISM_PYTHON" -c 'from huggingface_hub import snapshot_download; snapshot_download("deepseek-ai/DeepSeek-V2-Lite-Chat", revision="85864749cd611b4353ce1decdb286193298f64c7")'
bash run_stage.sh preflight --gpu
```

Pinned metadata includes `LICENSE`. A missing file, broken cache link or changed bytes stops verification. Do not remove the check. Download the file from the exact snapshot and verify its recorded hash. The model cache is not included in research backups.

## 5. Prepare once, then run the six-condition case

To generate answers without Judge calls or fitted centers:

```bash
export RUN=/workspace/experiment/runs/single_case_run01
cd "$CODE"
bash run_stage.sh prepare --input "$CODE/examples/benign_corpus.json" --corpus --run "$RUN" --study baseline
bash run_stage.sh run --run "$RUN" --until baseline
```

Use the supplied corpus wrapper so the example is recorded as `benign`; the legacy raw single-case loader retains its original group default for compatibility. This runs the native/capture smoke first. Smoke compares the new capture path to the original extractor and replays generation without the capture hook. Successful smoke generation is reused. The `baseline` boundary stops before Judge calls. For separate cache-disabled native extraction:

```bash
bash run_stage.sh run --run "$RUN" --until extract
```

Generation-time donor states use the actual first prefill with KV caching enabled. Separate native extraction uses `use_cache=False`. They are stored separately and are not silently substituted for one another.

One case has six baseline outputs. The verification replay adds six smoke-only generation forwards. Smoke replay outputs are evidence, not extra study samples. Extraction performs forward passes without answer decoding. More than one input case still uses the first case's six conditions for the initial smoke.

## 6. Private reference assets for full mechanism execution

Prepare a different run for a different protocol. A prepared baseline-only run cannot be silently changed into an intervention run because its protocol hash is frozen.

Use the private original restoration package to export its existing fit250 centers and unchanged Judge rubric. The importer verifies the known center bytes and original configuration/asset contracts. It does not fit new centers or execute source files.

```bash
export LEGACY_RESTORE_CODE=/path/to/private/deepseek_restore_v1.0.1
export PRIVATE_ASSETS=/workspace/experiment/private/reference_assets
bash run_stage.sh import-reference --source-dir "$LEGACY_RESTORE_CODE" --out "$PRIVATE_ASSETS"
```

For entity/purpose layer12, import the original shared readout artifact from the private purpose package:

```bash
export LEGACY_PURPOSE_CODE=/path/to/private/purpose_package
export ALIGNMENT_ASSETS=/workspace/experiment/private/alignment_assets
bash run_stage.sh import-alignment-reference --source-dir "$LEGACY_PURPOSE_CODE" --out "$ALIGNMENT_ASSETS"
```

Both imports have CPU-verified adapters. They require those private files to exist at the supplied paths. No C0 NPZ needs to be uploaded to a chat; the unified run produces and links its own fresh C0 donors. Existing original artifacts are read-only. It never replaces an unavailable donor with a mean, another question or an older unrelated cache.

For only the rubric:

```bash
bash run_stage.sh export-rubric --source "$LEGACY_RESTORE_CODE/vendor/r12/P10_judge.py" --out /workspace/experiment/private/rubric.json
```

A different externally fitted artifact can use `register-artifact`; it requires an explicit matching hash and evidence JSON with `kind`, `scope` and `sha256`. The command does not prove that a claimed fitting protocol was executed correctly. Prefer the verified original imports or artifacts produced by this package.

## 7. End-to-end mechanism run — RunPod SSH / tmux

```bash
tmux new-session -A -s mechanism
```

This section is for separately supplied research data and fitted artifacts. It is not a self-contained intervention demonstration on the bundled benign example. Inside that tmux shell, set the variables from sections 4 and 6 and `PRIVATE_CORPUS` to your explicit corpus file, then:

```bash
export RUN=/workspace/experiment/runs/mechanism_run01
cd "$CODE"
bash run_stage.sh prepare --input "$PRIVATE_CORPUS" --corpus --run "$RUN" --study all --centers "$PRIVATE_ASSETS/centers.npz" --centers-receipt "$PRIVATE_ASSETS/centers_receipt.json" --directions "$ALIGNMENT_ASSETS/directions.npz" --directions-receipt "$ALIGNMENT_ASSETS/directions_receipt.json" --rubric "$PRIVATE_ASSETS/rubric.json"
read -r -s -p 'Judge API key: ' RESEARCH_JUDGE_API_KEY
printf '\n'
export RESEARCH_JUDGE_API_KEY
mkdir -p "$RUN/logs"
set -o pipefail
bash run_stage.sh run --run "$RUN" --until complete 2>&1 | tee -a "$RUN/logs/pipeline.log"
```

Enter the key only in the RunPod Bash shell. It is read from the environment, not a source file. Never paste a key into a command argument or a document.

The graph is: smoke -> six baselines -> separate extraction -> optional cross baselines -> blinded baseline judging -> baseline gate -> immutable fresh RA -> intervention smoke -> interventions -> judging -> optional alignment restoration -> contemporary strength comparator rejudging -> behavioral/representation reports and data-eligible extended analyses.

Alternative `--study` values are `baseline`, `restore`, `strength`, `full-vector` and `alignment`. The default fresh RA membership uses valid same-run C0 explicit refusal and C4 full/partial compliance. Input exclusion, output failure, unclear judgments and missing artifacts are never interpreted as successful refusal restoration. This release stops on an overlength input; inspect or explicitly change the input in a new run rather than silently truncating or dropping it.

The 16 and 20 layer runs are separate; a response always starts from an unmodified recipient input and independent generation state. Each patch occurs once at the last prompt token in first prefill. The answer then decodes through the complete model. It does not patch each generated token or copy C0's KV cache.

| Family | New generation tasks per eligible item | Comparison |
|---|---:|---|
| restore | 6 | sham/restore/random × 16/20; alpha 1 |
| strength | 11 total, including the six above | additional layer16 restore/random at 1.5 and 2; HR-mean reference |
| full-vector | 2 | whole C0 vector at 16 or 20 |
| all, excluding alignment | 13 | union of restore, strength and full-vector |
| alignment | 8 per A-persona evaluation item | entity/purpose × four arms at layer12 |
| alignment with A/B cross templates | 16 per evaluation parent | A/B are paired users, not independent parents |

Intervention smoke tasks are included in these totals and reused after passing. Full-vector generation can cover the historical complete cohort by explicitly selecting the historical mode below. Random directions use one fixed draw per condition and match the corresponding actual post-BF16 restoration norm. They do not provide a size-matched control for full-vector replacement.

The shared layer12 direction is the original common congruence readout. Entity uses its same-question C3-E donor, purpose its C3-P donor. With B personas, BA is the donor for BB. Follow-up s at layers13–16 and f/g at16 are measured on the same affected forward pass. Entity restoration is newly implemented and has not been executed historically or here.

## 8. Larger private inputs, original cohorts and cross cells

`--corpus` accepts an object with a `cases` array. Each record has the exact single-case `input` object plus `metadata` containing `set_id`, `variant` (`harmful` or `benign`), `parent_question_id`, `shared_component_id`, and `dev_role`. Roles are `calibration_fit`, `readout_select`, `readout_check`, or an explicitly new role such as `new_evaluation`.

Question text and persona/documents are not rewritten. Metadata is supplied explicitly; the package does not infer domains, create benign questions, resplit original parents, or invent shared clusters. Duplicate case IDs within a set fail. Questions can repeat across set IDs and results are not automatically shared or pooled. Use one run per set when reproducing the historical sequential set workflow. No old workbook hash whitelist is imposed; the actual new input hash is frozen during prepare.

For the original historical-C0 cohort mode, add `historical_C0_label: "refusal"` and `historical_label_source_sha256` to harmful records, include matched benign records with the same parent, and prepare with `--cohort-mode historical`. This retains the explicitly supplied historical cohort for generation while reporting fixed fresh RA separately. Fresh C0 changes do not delete historical members. Historical labels must come from the actual evidence, not synthetic replacements for missing data.

Cross experiments additionally require `--cross /private/cross.json` at prepare. Its exact schema is `{"templates":{"SET:VARIANT:QUESTION_ID":{"entity_B":"explicit original B persona","purpose_B":"explicit original B persona"}}}`. The original B persona text is required; it is not inferred from a document. The bundled input remains unchanged. A/B cross variants are additional configured inputs and tasks.

## 9. Saved-result analysis, including local Mac execution

These commands read existing saved data, do not load model weights, and do not call a Judge. Use a new output path every time. For local use, set `MECHANISM_PYTHON` to the CPU environment and `RUN` to the downloaded run directory. Absolute activation paths in exported ledgers must point to the local extracted run; export a new local ledger after moving a run.

```bash
bash run_stage.sh analyze --run "$RUN" --out "$RUN/analysis_review01"
bash run_stage.sh export-ledger --run "$RUN" --out "$RUN/ledger_review01.jsonl"
bash run_stage.sh score --ledger "$RUN/ledger_review01.jsonl" --centers "$RUN/centers.npz" --directions "$RUN/directions.npz" --out "$RUN/scores_review01.jsonl"
bash run_stage.sh linkage --scores "$RUN/scores_review01.jsonl" --out "$RUN/linkage_review01"
bash run_stage.sh common-cohort --scores "$RUN/scores_review01.jsonl" --out "$RUN/common_review01"
bash run_stage.sh readouts --scores "$RUN/scores_review01.jsonl" --out "$RUN/readout_review01.csv"
bash run_stage.sh alignment-report --run "$RUN" --scores "$RUN/scores_review01.jsonl" --out "$RUN/alignment_review01"
```

If a direction artifact was not used, omit `--directions` and the alignment report. Original centers with t_A only produce f/g only at t_A; other scores remain null. All-position f/g needs the actual fit-derived center for each position. Raw activations are still recorded at all defined positions.

For strength results use `analyze ... --batch strength_contemporary` and `export-ledger ... --batch strength_contemporary`. This compares new responses against C4 and alpha1 responses newly judged in the same batch. `--batch main` preserves the earlier-label sensitivity. Fresh RA stays frozen from the original baseline batch; contemporaneous C4 labels may no longer all be compliant.

On a sufficient private corpus, these analysis functions are also available:

```bash
bash run_stage.sh validate-f --ledger "$RUN/ledger_review01.jsonl" --out "$RUN/split_validation01.json"
bash run_stage.sh validate-f --ledger "$RUN/ledger_review01.jsonl" --scope full500 --positions t_U t_D t_C t_A --out "$RUN/full_refit01.json"
bash run_stage.sh fit-centers --ledger "$RUN/ledger_review01.jsonl" --scope fit250 --out "$RUN/new_fit_centers01.npz"
bash run_stage.sh fit-centers --ledger "$RUN/ledger_review01.jsonl" --scope full500 --out "$RUN/full_centers01.npz"
bash run_stage.sh center-sensitivity --ledger "$RUN/ledger_review01.jsonl" --out "$RUN/center_sensitivity01"
bash run_stage.sh probe-analysis --ledger "$RUN/ledger_review01.jsonl" --centers "$RUN/centers.npz" --out "$RUN/probes01"
bash run_stage.sh regression --scores "$RUN/scores_review01.jsonl" --out "$RUN/regression01"
bash run_stage.sh alerts --scores "$RUN/scores_review01.jsonl" --mode c4 --out "$RUN/alerts_c4_01.json"
bash run_stage.sh alerts --scores "$RUN/scores_review01.jsonl" --mode delta --out "$RUN/alerts_delta_01.json"
```

A full500 refit is explicitly same-sample descriptive reselection; it is not a new independent check. Its bootstrap refits and evaluates using the **same** component draw. The original split validation uses independent role draws and never chooses a candidate on check data. Primary candidates remain t_C/t_A even when all positions are calculated. BR absence remains an H/R limitation. Full-only sensitivity retains selected layers. The standalone routines do not silently replace the intervention center bank.

`extended-analysis --run ... --ledger ... --scores ... --out ...` connects eligible saved-data recipes. `--study all` invokes it automatically. Missing paired benign data, roles or classes are explicitly recorded. Adjusted regression uses original logistic settings, topic controls, fold-only scaling, LOGO and select-to-check evaluation, 2000 component draws and 27/81-curve simultaneous bands. Alert thresholds use select negatives and strict `score > threshold`; they are not optimized on check. Alert flags are not behavioral refusals.

## 10. Metrics and uncertainty

- `f = cos(h, HR) - cos(h, BA)`; `g = cos(h, HA) - cos(h, HR)`; `a = cos(h, HA) - cos(h, BA)`.
- A restoration changes only the component along `normalize(normalize(HA)-normalize(HR))`; alpha1 is not whole-vector replacement.
- Alpha1.5 is the primary strength comparison against contemporaneously judged alpha1; alpha2 and HR mean are secondary.
- `strict_hcr` retains the original executable convention: valid clarification is zero compliance, not refusal. `paper_hcr` additionally excludes clarification. They are separate columns.
- Strict validity excludes truncation, unconfirmed termination, empty/protocol-only outputs, failed or absent judging, and undecided labels. Legacy HCR is retained separately.
- Explicit refusal recovery counts only a valid refusal. Missingness bounds, valid-pair denominators, fixed-cohort recovery denominators and unknown counts are preserved.
- Restoration: 2000 paired shared-component draws, seed20261102. The two layer-wise primary alpha1/full-vector comparisons have Bonferroni 97.5% intervals and separately calculated pointwise95% intervals. Strength uses pointwise95% intervals. Reduction is minus change; interval endpoints are reversed when signs are changed.
- Alignment/cross: original exact cluster resampling when at most four components, otherwise4000 draws seed20261108. Original alignment reports preserve A/B balancing, common-eight-output eligibility, pairwise eligibility, and follow-up recovery ratios. Near-zero natural gaps or intervals spanning zero suppress ratios. Generic 2000-draw alignment rows in the broad behavioral report are marked auxiliary; the original alignment report is authoritative for that family.
- Probes: all4^4 draws with four components, otherwise2000 draws seed1707; direct and shared direction intervals are conditional on fitted directions.
- A shift in f/g or s does not itself establish causal specificity, an independent H/R axis, an optimal causal layer, or response safety.

## 11. Status, interruption, resume and Judge errors

```bash
bash run_stage.sh status --run "$RUN"
tail -n 50 "$RUN/logs/pipeline.log"
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
```

Detach tmux with Ctrl-b then d. Reattach with `tmux attach -t mechanism`. `status` reads saved records and does not launch a stage or prove that an old `running` PID is alive.

Ctrl-C interrupts the foreground pipeline. Completed, hash-verified outputs and completed Judge payloads are reused when the same run command is repeated. A partially written uncommitted target can require one new generation. A recorded failed target is not automatically retried. A crash during a Judge request has an uncertain provider-side outcome; the transport journals it and requires explicit acknowledgment before a bounded retry, which can cause duplicate provider processing.

```bash
bash run_stage.sh run --run "$RUN" --until complete
```

For a revoked key, set a new environment value in the actual tmux shell. The already-running child process does not inherit a later shell export. After stopping or observing the failed process, inspect the failed record and archive only that failed payload with its task ID:

```bash
read -r -s -p 'New Judge API key: ' RESEARCH_JUDGE_API_KEY
printf '\n'
export RESEARCH_JUDGE_API_KEY
bash run_stage.sh retry-judge --run "$RUN" --batch main --task-id FAILED_TASK_ID
bash run_stage.sh run --run "$RUN" --until complete
```

Failed attempts are archived, never erased. A successful label or frozen baseline binding cannot be reset by this command. `judge --resume-uncertain` is the explicit interrupted-request path; it does not add an unlimited retry budget. Authentication, permission, billing and parser failures stop the sequence instead of calling the next question.

## 12. Optional diagnostics and audits

Numeric diagnostics are not automatically repeated every GPU run. They do not relax thresholds or reuse a historical manual waiver.

```bash
bash run_stage.sh prefix-audit --run "$RUN" --out "$RUN/prefix_audit_review01.json"
bash run_stage.sh audit-npz --path "$RUN/activations" --out "$RUN/npz_audit01.json"
bash run_stage.sh diagnose --run "$RUN" --mode repeat --out "$RUN/repeat_diagnostic01"
bash run_stage.sh diagnose --run "$RUN" --mode layer0 --out "$RUN/layer0_diagnostic01"
bash run_stage.sh diagnose --run "$RUN" --mode o-proj --out "$RUN/oproj_diagnostic01"
```

The first two are saved-file CPU checks. `diagnose` loads the GPU model and performs forward passes. Reference L2/cosine thresholds are reported diagnostically and are not validated DeepSeek acceptance rules.

`human-review` exports blinded question/response rows, or imports a completed review using `--review-file`. Technical error is only allowed for a genuinely technical/empty record. Neither human agreement nor job completion is automatic Judge quality validation.

To measure label drift on exactly the same response text, run `judge --batch NEW_BATCH_NAME`, then `temporal-report --first main --second NEW_BATCH_NAME`. A new batch makes new paid Judge calls. The temporal report itself is CPU-only and never changes prior labels or fresh RA membership. Null backend fingerprints are unknown.

The `import-cache` command accepts the original identity-indexed FP32 cache with an explicit semantic metadata JSONL. It verifies original cache and source NPZ hashes and rejects duplicate or missing IDs. See `run.py import-cache --help`. It does not assume that a row number identifies a question or reconstruct unprovided labels.

## 13. Full backup — RunPod, then download — Mac

After all writers exit, use one archive for the source, every output, activations, extraction arrays, Judge journals, logs, references copied into the run, and analysis files:

```bash
export BACKUP=/workspace/experiment/mechanism_run01_backup.tar.gz
bash run_stage.sh backup --run "$RUN" --out "$BACKUP"
```

The command refuses an existing destination and detects file changes during the snapshot. The backup is private. External original r12 inputs and model weights are not duplicated; preserve their original backups separately.

In the **Mac local terminal**, use the connection variables from section3:

```bash
mkdir -p "$HOME/Downloads/mechanism_run01_backup"
scp -i "$RUNPOD_SSH_KEY" -P "$RUNPOD_PORT" "root@$RUNPOD_HOST:/workspace/experiment/mechanism_run01_backup.tar.gz" "root@$RUNPOD_HOST:/workspace/experiment/mechanism_run01_backup.tar.gz.sha256" "root@$RUNPOD_HOST:/workspace/experiment/mechanism_run01_backup.tar.gz.inventory.json" "$HOME/Downloads/mechanism_run01_backup/"
cd "$HOME/Downloads/mechanism_run01_backup"
shasum -a 256 -c mechanism_run01_backup.tar.gz.sha256
open "$HOME/Downloads/mechanism_run01_backup"
```

The tar contains `run/` and `source/`. Verify the local hash before deleting remote research storage. Backup includes the private rubric and generated text; do not publish it as the public code package.

## 14. Validation and change history

See `verification/cpu_validation.json`, `verification/source_preservation.json` and `verification/anonymity_audit.json` for actual checks. CPU tensors/tokenizers in tests are explicit doubles. No synthetic result is represented as a real DeepSeek result.

Version1.0.0 consolidates the retained logic into a new directory and schema without modifying source archives. That parent release preserved its selected input bytes and native/numerical functions; this patch intentionally replaces only the public example and associated documentation/verification. New work includes the unified graph, external private-artifact adapters, English schema, and entity12 extension. GPU smoke and provider API validation remain unexecuted in the implementation environment.


### 1.0.1 changes (2026-11-18)

- Replaced the public example with a benign invitation task, preserving all six condition definitions. Added an explicit benign corpus wrapper without changing the input loader or research selection rules.
- Refreshed checksums and CPU/publication validation; the package version is 1.0.1 and the unchanged internal run/result schema remains 1.0.0.
- The fitted centers, directions, original private rubric, and fresh-RA gates remain required for the research modes. A benign case cannot demonstrate harmful-compliance reduction or guarantee any eligible restoration tasks.
- This release is included in `llm_research_public_v1.0.0_2026-11-18`; no original archive or completed run was modified.
