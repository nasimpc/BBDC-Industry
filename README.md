# Bremen Big Data Challenge 2026 — Solution

## Dual-Agent Architecture for Multi-Label SDG Classification

This repository contains my best solution for the **Bremen Big Data Challenge (BBDC) 2026**: multi-label classification of **17 UN Sustainable Development Goals (SDGs)** in German political documents using a **Dual-Agent Architecture**.

---

## Table of Contents

- [Dual-Agent Architecture](#dual-agent-architecture)
- [Agent 1: The Analyst Agent (The Ruleset Generator)](#agent-1-the-analyst-agent-the-ruleset-generator)
- [Agent 2: The Classifier Agent (The Forecaster)](#agent-2-the-classifier-agent-the-forecaster)
- [Robust 2-Fold Validation](#robust-2-fold-validation)
- [Architecture Diagram](#architecture-diagram)
- [How to Reproduce](#how-to-reproduce)
- [Configuration](#configuration)
- [Output Files](#output-files)
- [External Datasource & Data Modifications](#external-datasource--data-modifications)

---

## Dual-Agent Architecture

Rather than a monolithic pipeline, this solution decomposes the classification problem into two specialized agents that collaborate in sequence. 

| | Agent 1: Analyst | Agent 2: Classifier |
|---|---|---|
| **Role** | The Ruleset Generator | The Forecaster |
| **Input** | 100 labeled training documents | Knowledge artifacts + 1,400 unlabeled documents |
| **Output** | Reference Library, Ruleset, Thresholds | Final SDG predictions (CSV) |
| **LLM Mode** | Thinking Level: MEDIUM, temp 0.3 | Thinking Level: HIGH, temp 0.4 |
| **Runs** | Once per session | Per-document |

---

## Agent 1: The Analyst Agent (The Ruleset Generator)

The Analyst Agent is responsible for **distilling the entire training corpus into compact, actionable knowledge**. It does not classify documents directly — instead, it produces three knowledge artifacts that the Classifier Agent consumes.

### 1.1 Reference Library Selection (`select_reference_library`)

A **greedy set-cover algorithm** selects the minimal subset of training documents (~20–25) that collectively cover all 17 SDGs:

- **Phase 1 — Rare SDG Priority:** For each rare SDG (frequency ≤ 10 in training data), select the document that covers the most uncovered SDGs. This ensures rare goals (SDG2, SDG5, SDG6, SDG14, SDG15, SDG17) are represented first.
- **Phase 2 — Greedy Coverage:** Iteratively pick the document that covers the most remaining uncovered SDGs until all 17 are represented.
- **Phase 3 — Negative Example:** Include a zero-label document (the shortest available) as a negative example to improve precision and teach the model what "no SDGs" looks like.

**Result:** ~25 fully labeled documents with their complete text, covering all 17 SDGs with minimal redundancy.

### 1.2 Classification Ruleset Generation (`generate_ruleset`)

The remaining ~75 training documents (not in the Reference Library) are fed to the LLM to **generate a comprehensive classification guide** (the "micro-ruleset"). The Analyst specifically instructs the LLM to produce:

1. **Rare SDG Detailed Analysis (60% of output):** For SDG2, SDG5, SDG6, SDG14, SDG15, SDG17 — exhaustive German keyword lists, context patterns, trigger explanations, and negative indicators
2. **Confusing SDG Pair Disambiguation:** Rules to distinguish SDG7 vs SDG13, SDG9 vs SDG11, SDG3 vs SDG8, SDG6 vs SDG14
3. **Multi-Label Co-occurrence Patterns:** Common label combinations with concrete examples
4. **Zero-Label Indicators:** Patterns for documents addressing no SDGs
5. **Hard Case Rules:** Specific distinguishing criteria for SDG3 and SDG8

The ruleset is cached to disk (`micro_ruleset.md`) to avoid regeneration on subsequent runs.

### 1.3 Per-SDG Threshold Tuning (`tune_thresholds`)

Instead of a fixed 0.5 majority threshold, the Analyst performs a **grid search** over candidate thresholds `[0.0, 0.32, 0.34, 0.5, 0.66, 0.67, 1.0]` per SDG to find the threshold that maximizes F1-score on validation data. The tuned thresholds are saved to `optimal_thresholds.json`.

---

## Agent 2: The Classifier Agent (The Forecaster)

The Classifier Agent is a **pure inference engine**. It does not learn from training data directly — all knowledge is received from the Analyst Agent via the Reference Library, Ruleset, and Thresholds.

### 2.1 Prompt Construction

The prompt is structured in two parts:

**Static Prefix (cached, ~100k+ tokens):**
1. System instructions with all 17 SDG definitions
2. Rare SDG detection guidance with hardcoded German keyword lists for SDG6, SDG15, SDG17
3. Full-text Reference Library with labeled examples
4. Generated Classification Ruleset (disambiguation rules)

**Dynamic Suffix (per-document):**
5. Target document text
6. Explicit 6-point checklist forcing the model to verify rare SDG triggers before answering

### 2.2 Context Caching

The static prefix is uploaded once via Gemini's `CachedContent` API and reused across all 1,400+ predictions. This reduces per-call token costs by **>90%** and avoids re-processing ~100k tokens for every document.

### 2.3 Self-Consistency Majority Voting

Each document receives **3 independent predictions** (temperature=0.4, thinking_level=HIGH). The votes are averaged per SDG:

```
Vote Average(SDG_j) = (Vote1(SDG_j) + Vote2(SDG_j) + Vote3(SDG_j)) / 3
```

The averages are then binarized using the per-SDG thresholds from the Analyst Agent.

### 2.4 Historical Vote Injection

A previous submission's predictions (`submission_old.csv`) can be injected as one of the voting rounds, reducing API calls from 3→2 per document while maintaining ensemble diversity. This cuts inference cost by ~33%.

### 2.5 Robustness Features

- **Exponential backoff** for API rate limiting (HTTP 429 / RESOURCE_EXHAUSTED) — up to 5 retries with escalating waits (30s → 150s)
- **Checkpointing** every 50 documents to CSV + automatic Google Drive backup
- **Resume capability** via `RESUME_FROM_DOC` flag to skip already-processed documents after interruptions
- **Graceful fallback** to non-cached mode if context cache creation fails

---

## Robust 2-Fold Validation

The 100 labeled documents (train dataset) are split into two equal halves (50/50):

```
┌─────────────────────────────────────────────────────────────────┐
│                     FOLD 1                                      │
│                                                                 │
│  Analyst Agent trains on docs 0–49:                             │
│    → Builds Reference Library from docs 0–49                    │
│    → Generates Ruleset from remaining docs in 0–49              │
│                                                                 │
│  Classifier Agent tests on docs 50–99:                          │
│    → Uses Fold 1's Reference Library + Ruleset                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     FOLD 2                                      │
│                                                                 │
│  Analyst Agent trains on docs 50–99:                            │
│    → Builds Reference Library from docs 50–99                   │
│    → Generates Ruleset from remaining docs in 50–99             │
│                                                                 │
│  Classifier Agent tests on docs 0–49:                           │
│    → Uses Fold 2's Reference Library + Ruleset                  │
└─────────────────────────────────────────────────────────────────┘
```

### After Both Folds Complete

1. **Per-fold metrics** are reported (Macro F1, Precision, Recall per SDG)
2. **Combined metrics** across all 100 documents using majority vote (threshold=0.5)

The performance of the solution on validation runs was consistent with leaderboard scores, which validated the robustness of the cross-validation setup. This gave high confidence in each submission, allowing us to complete the competition with only **4 total submissions**.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│              AGENT 1 — The Analyst Agent                    │
│              (The Ruleset Generator)                        │
│                                                             │
│  100 labeled docs ─┬─► Reference Library (25 docs)         │
│                    │   (greedy set-cover, rare SDG prio)    │
│                    │                                        │
│                    └─► Remaining docs (~75) ─► LLM ──►     │
│                         Classification Ruleset              │
│                         (micro_ruleset.md)                  │
└────────────────────────────┬────────────────────────────────┘
                             │ Knowledge Artifacts
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              AGENT 2 — The Classifier Agent                 │
│              (The Forecaster)                               │
│                                                             │
│  Static Prompt Prefix (cached via Gemini API):              │
│  ┌──────────────────────────────────────────┐               │
│  │ 1. System Instructions + SDG Definitions │               │
│  │ 2. Rare SDG Detection Guidance           │               │
│  │ 3. Reference Library (full-text examples)│               │
│  │ 4. Classification Ruleset                │               │
│  └──────────────────────────────────────────┘               │
│                      ▼                                      │
│  For each test doc (1,400 docs):                            │
│  ┌──────────────────────────────────────────┐               │
│  │ 5. Target Document + Rare SDG Checklist  │               │
│  └──────────────────────────────────────────┘               │
│       │        │        │                                   │
│       ▼        ▼        ▼                                   │
│    Vote 1   Vote 2   Vote 3  (3 independent API calls)     │
│       │        │        │                                   │
│       └────┬───┘────────┘                                   │
│            ▼                                                │
│    Majority Vote Aggregation                                │
│            ▼                                                │
│    Final Binary Predictions                                 │
└─────────────────────────────────────────────────────────────┘
```

## How to Reproduce

### Step 1: Configure the Script

Edit the configuration flags at the top of `duel_agent.py`:

```python
VALIDATION_MODE = False  # Set True for cross-validation, False for final prediction
NUM_VOTES = 3            # Number of voting rounds per document
RESUME_FROM_DOC = 0      # Set to 0 to process all documents
HISTORICAL_VOTE_FILE = "submission_old.csv"  # Set to None to disable
```

### Step 2: Run Validation (Recommended First)

To evaluate on the training set with robust 2-fold cross-validation:

```python
VALIDATION_MODE = True
```
```bash
python duel_agent.py
```

This produces:
- `validation_results.md` — Per-SDG F1, precision, recall for both folds + combined metrics
- `optimal_thresholds.json` — Tuned per-SDG decision thresholds
- `val_predictions.csv` — Validation predictions

### Step 3: Run Final Predictions

```python
VALIDATION_MODE = False
```
```bash
python duel_agent.py
```

This produces:
- `submission_majority.csv` — Predictions using standard majority vote (threshold=0.5)
- `submission_tuned.csv` — Predictions using tuned per-SDG thresholds from `optimal_thresholds.json`

### Step 4: Submit

Upload `submission_tuned.csv` (or `submission_majority.csv`) to the BBDC 2026 Submission Portal.

---

## Configuration

| Flag | Default | Description |
|---|---|---|
| `VALIDATION_MODE` | `False` | `True` = 2-fold cross-validation on training set; `False` = full prediction on test set |
| `NUM_VOTES` | `3` | Number of independent LLM predictions per document for majority voting |
| `DRIVE_BACKUP_DIR` | `/content/drive/MyDrive/Colab` | Google Drive path for automatic checkpoint backups (Colab environment) |
| `RESUME_FROM_DOC` | `0` | Skip documents with ID < this number (for resuming interrupted runs) |
| `HISTORICAL_VOTE_FILE` | `submission_old.csv` | Path to previous submission CSV to inject as one voting round; set to `None` to disable |

---

## Output Files

| File | Description |
|---|---|
| `submission_majority.csv` | Final predictions using standard majority vote (threshold ≥ 0.5) |
| `submission_tuned.csv` | Final predictions using per-SDG optimized thresholds |
| `optimal_thresholds.json` | Per-SDG decision thresholds learned from validation |
| `micro_ruleset.md` | Learned information from train dataset (cached) |
| `validation_results.md` | Detailed validation metrics (only in validation mode) |
| `val_predictions.csv` | Validation predictions (only in validation mode) |

---

## External Datasource & Data Modifications

No external datasets were used. However, the original training data was lightly modified based on findings from data exploration:

- **Annotated example integration:** During exploration, a significant number of SDG 3 example cases were found in the training set, yet the model consistently struggled to predict this class. To address this, several training samples were replaced with **annotated examples that include explicit reasoning**, making the label signal clearer for the model. Additional high-quality cases were sourced from the provided annotated examples file and merged into the training set to improve rare-class coverage.
- **Modified training data:** The updated training dataset is included in this repository as **`train.zip`**.

- **Note on model selection:** The initial submission — which briefly held first place on the leaderboard — used **GPT-OSS 120B**. The final and best-performing submission switched to **Gemini 3.1 Pro**, with many new additional ideas, a natural evolution of first solution.

Thank you for organizing this competition, we learned a lot and it made our summer gap fun and productive.