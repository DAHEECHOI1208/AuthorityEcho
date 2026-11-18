# Public LLM evaluation and mechanism bundle

Version 1.0.0 | 2026-11-18

This archive combines two versioned source packages. Both use the same fictional, benign book-club invitation example. They retain separate execution engines, run directories, model settings, and Judge contracts; packaging them together does not make their model outputs or rubrics interchangeable.

| Directory | Purpose | Execution environment |
|---|---|---|
| `llm_evaluation_public_v1.4.2_2026-11-18` | Commercial target APIs followed by the configured four Judge profiles | Mac local terminal or RunPod; API/endpoint access required |
| `deepseek_mechanism_public_v1.0.1_2026-11-18` | Native DeepSeek generation, activation extraction, and research intervention/analysis implementation | CPU checks on Mac; real DeepSeek inference on the pinned CUDA environment |

The evaluation package uses its public English compliance rubric. The mechanism package retains the original external rubric requirement for historical compatibility. No private rubric, research corpus, historical responses, fitted centers, activation vectors, credentials, or original source archives are included.

## 1. Verify the downloaded source - Mac local terminal

Place the ZIP in Downloads, then run:

```bash
export BUNDLE=llm_research_public_v1.0.0_2026-11-18
cd "$HOME/Downloads" &&
test ! -e "$BUNDLE" &&
unzip "$BUNDLE.zip" &&
cd "$BUNDLE" &&
python3 -B verify_release.py
```

The verifier checks every declared file hash, rejects undeclared files, scans decoded source/JSON strings for Hangul and credential literals, and checks that the two inputs and explicit benign metadata agree. It runs no model and makes no network requests. The included verification reports distinguish actual CPU/mock checks from unexecuted live GPU/API checks.

The ZIP contains source code directly, not nested old ZIPs. Keep the source tree separate from private run directories and downloaded assets. To publish this release, use the verified source tree; do not add generated outputs or an external tokenizer cache.

## 2. Evaluation setup and execution - Mac local terminal

```bash
export BUNDLE_ROOT="$HOME/Downloads/llm_research_public_v1.0.0_2026-11-18"
export EVAL_CODE="$BUNDLE_ROOT/llm_evaluation_public_v1.4.2_2026-11-18"
python3.12 -m venv "$HOME/.venvs/llm-evaluation-public"
export EVAL_PYTHON="$HOME/.venvs/llm-evaluation-public/bin/python"
cd "$EVAL_CODE" &&
"$EVAL_PYTHON" -m pip install -r requirements.txt &&
"$EVAL_PYTHON" -B -m unittest discover -s tests -v
```

For HarmBench, explicitly fetch its unchanged pinned tokenizer into an external cache:

```bash
export LLM_HARMBENCH_TOKENIZER_CACHE="$HOME/.cache/llm-research-public/harmbench/bda705349d1144fa618770bea64d99ce54e3835b"
cd "$EVAL_CODE" &&
"$EVAL_PYTHON" -B setup_harmbench_tokenizer.py download &&
"$EVAL_PYTHON" -B setup_harmbench_tokenizer.py check
```

These setup commands download four public resource files, with revision, size, and SHA256 verification. They do not generate answers or call Judges. Ordinary execution only reads the validated cache and will stop if it is missing or changed. No vocabulary entries or token IDs are altered. The multilingual third-party resources are excluded from this public ZIP and must stay outside the source tree. Repeating `download` reuses verified files.

Set API credentials outside the source tree. For an external environment file:

```bash
mkdir -p "$HOME/.config/llm-research-public"
test ! -e "$HOME/.config/llm-research-public/evaluation.env" &&
cp "$EVAL_CODE/judges/configs/env.example" "$HOME/.config/llm-research-public/evaluation.env" &&
chmod 600 "$HOME/.config/llm-research-public/evaluation.env"
```

Edit that file locally and replace the credential and endpoint placeholders. Never upload it. The four Judge profiles require the accounts/endpoints listed in the evaluation README; provider access and live compatibility have not been verified here.

To run all six target profiles and four Judges after setup:

```bash
export EVAL_RUN="$HOME/llm-research-runs/evaluation_example_01"
cd "$EVAL_CODE" &&
"$EVAL_PYTHON" -B pipeline.py run --out "$EVAL_RUN" --env "$HOME/.config/llm-research-public/evaluation.env"
```

This command makes paid provider calls: 36 target-response slots and up to 144 main Judge calls before retries or separately invoked smoke checks. To run only one configured target and one Judge, instead start a new run with:

```bash
"$EVAL_PYTHON" -B pipeline.py run --out "$HOME/llm-research-runs/evaluation_small_01" --env "$HOME/.config/llm-research-public/evaluation.env" --target-model gpt-4.1-2025-04-14 --judges gpt56_luna_research
```

Resume the all-profile run with:

```bash
cd "$EVAL_CODE" &&
"$EVAL_PYTHON" -B pipeline.py resume --out "$EVAL_RUN" --env "$HOME/.config/llm-research-public/evaluation.env"
```

Reports are under `$EVAL_RUN/judge/`. Read the evaluation README for explicit failed/unknown request retry options. Source/config/input changes require a new run. Older runs must be resumed with their original package.

## 3. Upload and connect - Mac local terminal

Set the SSH values from your own RunPod Connect screen. These are placeholders, not a stored host identity.

```bash
export BUNDLE=llm_research_public_v1.0.0_2026-11-18
export RUNPOD_HOST='YOUR_RUNPOD_SSH_HOST'
export RUNPOD_PORT='YOUR_RUNPOD_SSH_PORT'
export RUNPOD_SSH_KEY="$HOME/.ssh/id_ed25519"
ssh -i "$RUNPOD_SSH_KEY" -p "$RUNPOD_PORT" "root@$RUNPOD_HOST" 'mkdir -p /workspace/experiment/packages' &&
scp -i "$RUNPOD_SSH_KEY" -P "$RUNPOD_PORT" "$HOME/Downloads/$BUNDLE.zip" "root@$RUNPOD_HOST:/workspace/experiment/packages/" &&
ssh -i "$RUNPOD_SSH_KEY" -p "$RUNPOD_PORT" "root@$RUNPOD_HOST"
```

## 4. DeepSeek benign generation and extraction - RunPod terminal

```bash
export BUNDLE=llm_research_public_v1.0.0_2026-11-18
cd /workspace/experiment/packages &&
test ! -e "$BUNDLE" &&
unzip "$BUNDLE.zip" &&
cd "$BUNDLE" &&
python3 -B verify_release.py
```

Enter a persistent terminal session:

```bash
tmux new-session -A -s mechanism
```

Inside tmux, use the matching existing CUDA environment. The Python path below is a generic example and must point to your actual installation:

```bash
export BUNDLE_ROOT=/workspace/experiment/packages/llm_research_public_v1.0.0_2026-11-18
export CODE="$BUNDLE_ROOT/deepseek_mechanism_public_v1.0.1_2026-11-18"
export MECHANISM_PYTHON=/workspace/experiment/.venv/bin/python
export HF_HOME=/workspace/experiment/model-cache
export RUN=/workspace/experiment/runs/benign_example_01
cd "$CODE" &&
"$MECHANISM_PYTHON" -B -m unittest discover -s tests -v &&
bash run_stage.sh preflight --gpu
```

The exact model, tokenizer revision, BF16/attention/YaRN settings, token budget, and library versions remain those in the mechanism configuration. Do not upgrade or relax them merely to pass preflight. Its README provides the explicit pinned model-download command if the checkpoint is not cached.

Prepare the benign example with explicit metadata, then generate and extract:

```bash
cd "$CODE" &&
bash run_stage.sh prepare --input "$CODE/examples/benign_corpus.json" --corpus --run "$RUN" --study baseline &&
bash run_stage.sh run --run "$RUN" --until extract
```

This runs the original smoke checks, six-condition generation, and separate native activation extraction. It stops before Judge calls and needs no fitted centers or private Judge rubric. The same harmless question is used in all six conditions. C3-E changes the fictional organization; C3-P changes the purpose; C4 matches both.

The corpus wrapper labels the case `benign`. Do not substitute the raw single-case file into this command: the original raw-case loader's group default has been retained for research compatibility.

Resume without repeating preparation:

```bash
cd "$CODE" &&
bash run_stage.sh run --run "$RUN" --until extract
```

Reattach later using `tmux attach -t mechanism`. Successful committed generations are reused under the existing identity checks.

## 5. Intervention and scientific interpretation boundary

This is a content/anonymity release patch, not a new intervention algorithm. Original fresh-RA selection, fitted reference requirements, hook location, directions, strengths, controls, Judge rules, and statistical calculations are preserved. A benign example is not relabeled harmful or forced into fresh RA. It may have no eligible restoration tasks.

Full research intervention modes still require their external fixed centers/directions and original Judge rubric. Their commands remain in the mechanism README. This bundle does not fabricate those assets, weaken the gate, or claim that one benign example reproduces the paper's refusal-recovery findings. A fully independent benign intervention demonstration would be a separately specified mode.

On the evaluation example, compliance means normal benign-task fulfillment. Safety-Judge outputs on that case do not establish harmful-compliance rates or Judge accuracy. Synthetic smoke refusals are test fixtures and are not measured target responses.

## 6. Download the DeepSeek run - RunPod, then Mac

After execution has finished, in the RunPod terminal:

```bash
cd "$CODE" &&
bash run_stage.sh backup --run "$RUN" --out /workspace/experiment/benign_example_01_backup.tar.gz
```

This includes the source and full run, including activation files and logs. In the Mac local terminal, using the SSH variables from section 3:

```bash
mkdir -p "$HOME/Downloads/benign_example_01_results" &&
scp -i "$RUNPOD_SSH_KEY" -P "$RUNPOD_PORT" "root@$RUNPOD_HOST:/workspace/experiment/benign_example_01_backup.tar.gz" "root@$RUNPOD_HOST:/workspace/experiment/benign_example_01_backup.tar.gz.sha256" "$HOME/Downloads/benign_example_01_results/" &&
cd "$HOME/Downloads/benign_example_01_results" &&
shasum -a 256 -c benign_example_01_backup.tar.gz.sha256
```

The output archive is private working data, not a replacement public source release.

## 7. Release checks and change history

The `verification/` directory records the CPU/mock results, source/config preservation comparison, external resource identity check, and publication scan. The root checksum manifest covers all shipped files except itself. Required third-party license notices are retained.

- 1.0.0 (2026-11-18): combined mechanism 1.0.1 and evaluation 1.4.2; synchronized benign inputs; benign smoke fixtures; external pinned tokenizer setup; integer Unicode-range checks; refreshed documentation, source hashes, and CPU/static verification. Parent archives and completed experiments were preserved.

Live GPU generation, provider calls, and network tokenizer download were not run during release preparation. Cache byte identity was checked using the previously supplied pinned resource files; download/resume/failure handling was tested with mocked transport.
