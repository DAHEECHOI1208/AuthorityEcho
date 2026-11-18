# LLM evaluation example

Version 1.4.2 | 2026-11-18

This package generates fresh responses for one synthetic example and evaluates
those responses with four judge profiles. The example requests a welcoming
invitation to an adult neighborhood book club. The organizations, community
coordinator persona, and reference notes are fictional. The example demonstrates execution and
condition alignment. It provides no measured model performance claim.

Requires Python 3.12. Run commands from the directory containing pipeline.py.

Setup

    python3.12 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt
    cp judges/configs/env.example .env

Fill in the API credentials and endpoint URLs in .env.
Replace all YOUR- placeholders before running; judge connections reject them.
Keep credentials outside the public source tree. The optional --env argument
accepts the location of an external credential file.
OPENAI_API_KEY serves the OpenAI targets and the Luna judge.
ANTHROPIC_API_KEY and GEMINI_API_KEY serve their respective targets.
HF_TOKEN, HF_HARMBENCH_URL, and HF_LLAMA3_URL serve HarmBench and StrongREJECT.
TOGETHER_API_KEY serves the DeepSeek judge.
Model availability and account access depend on each provider.

External HarmBench tokenizer setup - Mac or RunPod

The multilingual tokenizer is deliberately not included in the public ZIP.
Its vocabulary, IDs, configuration and revision are unchanged. Before using
HarmBench, explicitly download and verify its four pinned files:

    python setup_harmbench_tokenizer.py download
    python setup_harmbench_tokenizer.py check

The default is an external user cache. Set LLM_HARMBENCH_TOKENIZER_CACHE to an
absolute directory outside this source tree to use another location. Export
the same variable for setup and execution. A missing or changed cache stops
HarmBench; normal execution never silently downloads a replacement. Setup
requests public resource files only, not target generation or Judge calls.
A download interrupted after a verified file resumes by reusing that file.
Changed cached files are rejected without overwrite; use a clean cache path.
The downloaded third-party vocabulary can contain multiple languages and must
not be copied back into the public source archive.

Input

input.json contains exactly one case object with question_id, question,
depth1, depth2, domain, persona, and documents. The documents object contains
c3_e, c3_p, and c4. These are variants for the same question. The runner rejects
an array of cases. All inputs needed to construct the requests are bundled.

C0 uses the question and an empty reference list.
C1 adds the persona and uses an empty reference list.
C2 uses the question and the c4 reference note.
C3-E adds the persona and the organization-mismatched c3_e note.
C3-P adds the persona and the purpose-mismatched c3_p note.
C4 adds the persona and the matching c4 note.

The reference note is serialized as a retrieve_reference tool result in the
provider's native message format. Each target request is independently built
from its condition. The judge receives the original question and the full
response text. Target errors, empty answers, and incomplete generations are
recorded with their exclusion states.

Evaluation language

The compliance judges use the English response_compliance_6class_en_v1.0
rubric and the labels compliant, partially_compliant, refusal,
clarification_request, and undetermined. API errors are recorded separately.
Rubric wording and output labels are part of the evaluation configuration.
Use a fresh output directory for this release so each run records the exact
rubric, code and settings used.

Execution

    python pipeline.py run --out runs/example_01

The target order is gpt-4.1-2025-04-14, gpt-5.1-2025-11-13,
gpt-5.4-2026-03-05, gpt-5.6-luna, claude-opus-4-6,
and gemini-3.1-flash-lite. Each target completes C0, C1, C2, C3-E,
C3-P, and C4 before the next target begins.

After all targets finish, judges run in this order.

    gpt56_luna_research
    harmbench_full
    strongreject
    deepseek_research

Each judge processes the eligible responses from all six targets in target
and condition order. The default run has 36 target-response slots and up to
144 judge calls before retries. HarmBench evaluates full response text within
its 2048-token context budget and records overlength inputs as excluded.

Optional arguments include --input for another single-case JSON file,
--target-model for one of the configured targets, --target-config for that
single target's native settings, and --judges for a subset of judge profiles.
For example, add --target-model gpt-4.1-2025-04-14
--judges gpt56_luna_research to run one target and one judge.

Use a new --out directory for a fresh generation. A partially completed run
can be continued with its recorded input, settings, and response identities.

    python pipeline.py resume --out runs/example_01

Before the judge snapshot is prepared, --retry-target-errors retries failed
target attempts. --retry-target-unknown explicitly retries requests whose
delivery is uncertain, which can duplicate an API charge.

For recorded judge errors, the judge entry point supports explicit retries.

    python judges/judge.py resume --run-dir runs/example_01/judge --retry-errors

Results

runs/example_01/targets/<model>/ contains target records and response files.
runs/example_01/judge/results.xlsx contains the combined evaluations.
runs/example_01/judge/logs/results.jsonl contains detailed judge results.
The input snapshot and per-request identifiers connect evaluations to the
freshly generated responses. Run output includes request and response content.
The included ignore rules cover default run folders, environments, and caches.

Offline verification

    python tests/test_pipeline.py -v

The tests use mock API responses and a mock tokenizer. They check all six
native target profiles, condition-specific persona/document alignment, four
judge parsers, interruption and resume, input snapshot integrity, and the
single-case requirement. Additional regression checks exercise all five
English compliance labels, the standalone workbook exporters, and credential
placeholder rejection before judge HTTP calls. They do not establish live
provider compatibility or judge accuracy. The supported user entry points are pipeline.py and
judges/judge.py. Files inside judges/legacy are imported implementation modules.

Third-party resources

The bundled HarmBench prompts and StrongREJECT templates retain their license
notices under judges/legacy/*/assets. HarmBench asset provenance is recorded
in judges/legacy/harmbench/assets/provenance.json. The source registry records
hashes of the modules included in this release.


## Interpretation and validation boundary

The example tests execution and condition construction. On this benign input,
compliance is normal task fulfillment, not harmful compliance. HarmBench and
StrongREJECT remain unchanged safety evaluators; their scores on a benign case
are not evidence of their accuracy or of refusal erosion. Synthetic smoke
answers are deliberate refusals to a benign request and are not target data.
No live provider compatibility or accuracy was verified in this release.

Run all offline checks, including external-cache tests:

```bash
python -B -m unittest discover -s tests -v
```

## Change history

- 1.4.2 (2026-11-18): replaced the bundled case and smoke request with benign text; aligned the synthetic smoke refusal; moved four unchanged pinned tokenizer resources to an explicitly prepared external cache; retained strict hash verification and local-only loading; replaced Hangul string-range checks with integer code-point checks; renamed this README and refreshed source hashes.
- Target settings, condition definitions, Judge rubrics/parsers, scoring, retry policies and statistical rules remain unchanged. Version and integrity guards require a fresh run for this release; resume older runs with their original package.
