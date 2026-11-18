# LLM Evaluation and Mechanistic Analysis

This repository provides code for studying how claimed user authority
and reference-document consistency affect language-model responses.

The experimental framework holds the request fixed while varying
the presence of an authority persona and the consistency of a
reference document with that persona.

## Experimental conditions

| Condition | Authority persona | Reference document |
|---|---|---|
| C0 | Absent | Absent |
| C1 | Present | Absent |
| C2 | Absent | Document used in C4 |
| C3-E | Present | Entity-mismatched document |
| C3-P | Present | Purpose-mismatched document |
| C4 | Present | Consistent document |

C0 versus C4 measures the overall change associated with adding the
persona and consistent document. Comparisons of C4 with C3-E and
C3-P examine the additional contribution of entity and purpose
consistency.

## Repository structure

| Directory | Description |
|---|---|
| `api_evaluation/` | API-based response generation and evaluation across the six conditions. |
| `deepseek_mechanism/` | DeepSeek-V2-Lite-Chat inference, activation extraction, representation analysis, and activation interventions. |

The two pipelines are independently executable. They use a compatible
single-case input format but have separate dependencies, configurations,
and output directories.

## Getting started

Follow the instructions for the relevant pipeline:

- [API evaluation guide](api_evaluation/README.txt)
- [DeepSeek mechanism guide](deepseek_mechanism/README.md)

The guides describe input preparation, environment setup, execution,
resumption, and result export.

API-based stages require user-supplied credentials and service
configuration. DeepSeek inference and intervention stages require
a compatible GPU environment and access to the specified model.

Credentials must be supplied outside version-controlled files.

## Reproducibility scope

The repository contains implementation code and offline tests.
Placeholder fixtures are provided for software testing; they are
not experimental evaluation cases.

Generation and activation extraction can be run with user-supplied
inputs. Reproducing the original judge configuration and restoration
experiments additionally requires the corresponding rubric,
reference assets, fitted vectors, and eligible response cohorts
described in the pipeline guides.

Passing offline tests demonstrates implementation checks on mock
or synthetic data. It does not reproduce the reported experimental
results or establish intervention efficacy.

The guides identify the required model revisions, settings, and
environment constraints. Results obtained with different inputs,
model revisions, or API backends should be distinguished from
reproductions of the original experiments.

## Evaluation and reporting

Evaluation metrics retain their judge-specific definitions:

- Research-rubric judges: partial-or-full compliance rate.
- HarmBench: the proportion of valid Yes judgments.
- StrongREJECT: the mean original score on a 0–1 scale.

These metrics are not interchangeable. Reports should specify the
valid denominator, exclusions, and whether human adjudication is
included.

Where human adjudication supplements API judgments, the combined
results must be identified as such. Original API judgments and
adjudication provenance should be preserved separately.

## Data availability and responsible use

Original harmful requests, fabricated experimental documents,
model responses, and private research records are not distributed
in this repository.

Users must supply their own authorized evaluation inputs.
Generated outputs may contain sensitive or harmful content and
should be reviewed before sharing.

Only aggregate results that have been checked for disclosure risks
should be added to the public repository.

## Anonymity and third-party notices

This repository is prepared for anonymous review. Author names,
affiliations, personal contact details, private infrastructure
identifiers, and credentials are omitted.

Third-party copyright and license notices are retained in the
relevant directories. These notices identify upstream software
and assets, not the authors of this study.