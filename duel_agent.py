"""
Dual-Agent Architecture for UN SDG Classification
==================================================

Bremen Big Data Challenge 2026 — Multi-Label SDG Classification

Architecture:
  1. The Analyst Agent (The Ruleset Generator):
     - Ingests all 100 labeled training documents
     - Selects a Reference Library via greedy set-cover (prioritising rare SDGs)
     - Generates a comprehensive Classification Ruleset from the remaining documents
     - Tunes per-SDG decision thresholds via 2-fold cross-validation
     → Outputs: Reference Library, micro_ruleset.md, optimal_thresholds.json

  2. The Classifier Agent (The Forecaster):
     - Receives the Reference Library + Ruleset as a cached context prefix
     - For each test document, runs N self-consistency voting rounds
     - Aggregates votes and applies tuned thresholds to produce final labels
     → Outputs: submission_majority.csv, submission_tuned.csv
"""

import os
import sys
import json
import time
import zipfile
import shutil
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import KFold

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Please install google-genai: pip install google-genai")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
VALIDATION_MODE = True  # Toggle: True for 2-split validation (50/50), False for Full Normal Prediction Mode
NUM_VOTES = 3            # Number of self-consistency voting runs per document
DRIVE_BACKUP_DIR = "/content/drive/MyDrive/Colab"  # Google Drive backup path
RESUME_FROM_DOC = 0    # Skip docs before this number (set to 0 or None to disable)
HISTORICAL_VOTE_FILE = "submission_old.csv"  # Use as Vote 1 (set to None to disable)

# SDGs definitions
SDGS_DEFS = """
1. No Poverty - Ending poverty in all its forms everywhere through access to basic resources and social security systems.
2. Zero Hunger - Ensuring food security, improving nutrition, and promoting sustainable agriculture for all people.
3. Good Health and Well-being - Ensuring healthy lives and promoting well-being for all people at all ages.
4. Quality Education - Ensuring inclusive, equitable, and quality education and promoting lifelong learning opportunities for all.
5. Gender Equality - Achieving gender equality and empowering all women and girls for self-determination.
6. Clean Water and Sanitation - Ensuring availability and sustainable management of water and sanitation for all.
7. Affordable and Clean Energy - Ensuring access to affordable, reliable, sustainable, and modern energy for all people.
8. Decent Work and Economic Growth - Promoting sustained, inclusive economic growth, productive full employment, and decent work for all.
9. Industry, Innovation and Infrastructure - Building resilient infrastructure, promoting inclusive and sustainable industrialization, and fostering innovation.
10. Reduced Inequalities - Reducing inequality within and among countries through targeted policy measures.
11. Sustainable Cities and Communities - Making cities and human settlements inclusive, safe, resilient, and sustainable for all residents.
12. Responsible Consumption and Production - Ensuring sustainable consumption and production patterns through efficient use of resources and waste reduction.
13. Climate Action - Taking urgent action to combat climate change and its impacts at a global level.
14. Life Below Water - Conserving and sustainably using the oceans, seas, and marine resources for sustainable development.
15. Life on Land - Protecting, restoring, and promoting sustainable use of terrestrial ecosystems and forests, and combating desertification.
16. Peace, Justice and Strong Institutions - Promoting peaceful and inclusive societies, access to justice for all, and building effective, accountable institutions.
17. Partnerships for the Goals - Strengthening the means of implementation and revitalizing the global partnership for sustainable development.
"""

SDG_COLS = [f"SDG{i}" for i in range(1, 18)]


# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────

def save_to_drive(local_path):
    """Copy a local file to Google Drive backup directory (if mounted)."""
    if os.path.exists(DRIVE_BACKUP_DIR):
        dest = os.path.join(DRIVE_BACKUP_DIR, os.path.basename(local_path))
        shutil.copy2(local_path, dest)
        print(f"  [Drive] Backed up to {dest}")


def get_gemini_client():
    PROJECT_ID = "bbdc-491514"
    if not PROJECT_ID or PROJECT_ID == "[your-project-id]":
        PROJECT_ID = str(os.environ.get("GOOGLE_CLOUD_PROJECT"))

    # Check if we are in Colab
    if "google.colab" in os.sys.modules:
        from google.colab import auth
        auth.authenticate_user()

    print(f"Initializing Vertex AI client for project {PROJECT_ID}...")
    try:
        client = genai.Client(vertexai=True, project=PROJECT_ID, location="global")
        return client
    except Exception as e:
        print(f"Vertex initialization failed: {e}. Falling back to standard API key.")
        return genai.Client()


def ensure_test_folder():
    if not os.path.exists("test") and os.path.exists("test.zip"):
        print("Unzipping test folder...")
        with zipfile.ZipFile("test.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
    if not os.path.exists("train") and os.path.exists("train.zip"):
        print("Unzipping train folder...")
        with zipfile.ZipFile("train.zip", 'r') as zip_ref:
            zip_ref.extractall(".")


def parse_json_response(text):
    text = text.strip()
    if text.startswith('```json'): text = text[7:]
    elif text.startswith('```'): text = text[3:]
    if text.endswith('```'): text = text[:-3]
    text = text.strip()
    try:
        return json.loads(text)
    except Exception as e:
        print(f"Error parsing JSON: {e}\nText was: {text[:100]}")
        return {f"SDG{i}": 0 for i in range(1, 18)}


def evaluate_metrics(y_true, y_pred):
    f1s, precisions, recalls = [], [], []
    for i in range(17):
        f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        p = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
        r = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
        f1s.append(f1)
        precisions.append(p)
        recalls.append(r)
    return np.mean(f1s), np.mean(precisions), np.mean(recalls), f1s, precisions, recalls


# ═════════════════════════════════════════════════════════════
# AGENT 1 — The Analyst Agent (The Ruleset Generator)
# ═════════════════════════════════════════════════════════════
#
# Responsibilities:
#   • Select a Reference Library from labeled training data
#   • Generate a comprehensive Classification Ruleset (micro-ruleset)
#   • Tune per-SDG decision thresholds via cross-validation
#
# The Analyst Agent distills the training corpus into compact,
# actionable knowledge that the Classifier Agent can consume.
# ═════════════════════════════════════════════════════════════

class AnalystAgent:
    """
    The Analyst Agent (The Ruleset Generator).

    Ingests labeled training documents and produces three knowledge artifacts:
      1. Reference Library  — a minimal set of labeled docs covering all 17 SDGs
      2. Classification Ruleset — an LLM-generated disambiguation & keyword guide
      3. Optimal Thresholds  — per-SDG decision boundaries tuned on validation data
    """

    def __init__(self, client, model_id, train_dir="train"):
        self.client = client
        self.model_id = model_id
        self.train_dir = train_dir

    # ── Reference Library Selection ──────────────────────────

    def select_reference_library(self, df, max_docs=25):
        """
        Greedy set-cover: select minimal docs to cover all 17 SDGs.
        Prioritizes rare SDGs first, then greedily picks docs covering
        the most remaining SDGs.  Returns (selected_docs_list, remaining_df).
        """
        # Count SDG frequencies to identify rare ones
        sdg_freq = {s: df[s].sum() for s in SDG_COLS}

        # Sort SDGs by frequency (rarest first) to prioritize coverage
        sdgs_by_rarity = sorted(SDG_COLS, key=lambda s: sdg_freq[s])

        remaining_sdgs = set(SDG_COLS)
        selected_ids = []

        # Phase 1: For each rare SDG (<=10 examples), pick the best doc
        for sdg in sdgs_by_rarity:
            if sdg not in remaining_sdgs:
                continue
            if sdg_freq[sdg] > 10:
                continue
            candidates = df[df[sdg] == 1]
            candidates = candidates[~candidates['doc_id'].isin(selected_ids)]
            if len(candidates) == 0:
                continue
            best_doc = None
            best_score = 0
            for _, row in candidates.iterrows():
                score = sum(1 for s in remaining_sdgs if row[s] == 1)
                if score > best_score:
                    best_score = score
                    best_doc = row['doc_id']
            if best_doc:
                selected_ids.append(best_doc)
                row = df[df['doc_id'] == best_doc].iloc[0]
                covered = [s for s in SDG_COLS if row[s] == 1]
                remaining_sdgs -= set(covered)
                print(f"  [Analyst/RefLib] Selected {best_doc}: covers {covered}")

        # Phase 2: Greedy set-cover for any remaining uncovered SDGs
        while remaining_sdgs and len(selected_ids) < max_docs:
            best_doc = None
            best_score = 0
            for _, row in df.iterrows():
                if row['doc_id'] in selected_ids:
                    continue
                score = sum(1 for s in remaining_sdgs if row[s] == 1)
                if score > best_score:
                    best_score = score
                    best_doc = row['doc_id']
            if best_doc is None or best_score == 0:
                break
            selected_ids.append(best_doc)
            row = df[df['doc_id'] == best_doc].iloc[0]
            covered = [s for s in SDG_COLS if row[s] == 1]
            remaining_sdgs -= set(covered)
            print(f"  [Analyst/RefLib] Selected {best_doc}: covers {covered}")

        if remaining_sdgs:
            print(f"  [Warning] Could not cover: {remaining_sdgs}")

        # Phase 3: Add a zero-label doc as a negative example
        zero_docs = df[df[SDG_COLS].sum(axis=1) == 0]
        zero_docs = zero_docs[~zero_docs['doc_id'].isin(selected_ids)]
        if len(zero_docs) > 0 and len(selected_ids) < max_docs:
            best_zero = None
            best_size = float('inf')
            for _, row in zero_docs.iterrows():
                fpath = os.path.join(self.train_dir, f"{row['doc_id']}.txt")
                if os.path.exists(fpath):
                    sz = os.path.getsize(fpath)
                    if sz < best_size:
                        best_size = sz
                        best_zero = row['doc_id']
            if best_zero:
                selected_ids.append(best_zero)
                print(f"  [Analyst/RefLib] Selected {best_zero}: ZERO-LABEL negative example ({best_size} chars)")

        # Build reference library data
        ref_docs = []
        for doc_id in selected_ids:
            row = df[df['doc_id'] == doc_id].iloc[0]
            fpath = os.path.join(self.train_dir, f"{doc_id}.txt")
            if os.path.exists(fpath):
                with open(fpath, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                text = row.get('text', '')
            labels = {s: int(row[s]) for s in SDG_COLS}
            active = [s for s in SDG_COLS if labels[s] == 1]
            ref_docs.append({
                'doc_id': doc_id,
                'text': text,
                'labels': labels,
                'active_sdgs': active
            })

        remaining_df = df[~df['doc_id'].isin(selected_ids)].reset_index(drop=True)

        total_chars = sum(len(d['text']) for d in ref_docs)
        print(f"  [Analyst/RefLib] Total: {len(ref_docs)} docs, ~{total_chars} chars, covering {17 - len(remaining_sdgs)}/17 SDGs")

        return ref_docs, remaining_df

    # ── Ruleset Generation ───────────────────────────────────

    def generate_ruleset(self, remaining_df, ref_doc_ids, cache_file):
        """
        Generate a comprehensive classification guide from remaining docs.
        Uses full document text (up to 6000 chars) for maximum context.
        Focuses heavily on rare SDGs (SDG2, SDG5, SDG6, SDG14, SDG15, SDG17).
        """
        if os.path.exists(cache_file):
            print(f"\n[Analyst] Found existing {cache_file}. Loading...")
            with open(cache_file, "r", encoding="utf-8") as rf:
                return rf.read()

        print(f"\n[Analyst] Generating comprehensive ruleset from {len(remaining_df)} supplementary documents...")

        # Identify rare SDG docs in the remaining set
        rare_sdgs = ['SDG2', 'SDG5', 'SDG6', 'SDG14', 'SDG15', 'SDG17']
        rare_docs = remaining_df[remaining_df[rare_sdgs].sum(axis=1) > 0]
        non_rare_docs = remaining_df[remaining_df[rare_sdgs].sum(axis=1) == 0]

        prompt = f"""You are an expert political analyst specializing in UN Sustainable Development Goals (SDGs).

I have a REFERENCE LIBRARY of {len(ref_doc_ids)} fully labeled example documents.
The reference library documents are: {', '.join(ref_doc_ids)}.

Below are {len(remaining_df)} ADDITIONAL training documents with their true labels.
Your task is to analyze these documents and produce a DETAILED classification guide.

The guide MUST contain (in order of priority):

1. **RARE SDG DETAILED ANALYSIS** (MOST IMPORTANT - dedicate 60% of your output to this):
   For each rare SDG (SDG2, SDG5, SDG6, SDG14, SDG15, SDG17):
   - List ALL specific German keywords, phrases and vocabulary that triggered that SDG label
   - Describe the document context patterns in detail  
   - Explain WHY this document was labeled with that SDG, connecting specific text passages to the SDG definition
   - List negative indicators (what looks similar but does NOT trigger this SDG)
   
2. **Confusing SDG Pairs**: Detailed rules to distinguish commonly confused SDG pairs (e.g., SDG7 vs SDG13, SDG9 vs SDG11, SDG3 vs SDG8, SDG6 vs SDG14). Include concrete German keywords.

3. **Multi-Label Co-occurrence Patterns**: Common combinations with specific examples.

4. **Zero-Label Indicators**: Patterns for documents with NO SDGs.

5. **Hard Cases for SDG3 and SDG8**: Specific distinguishing rules.

IMPORTANT: For rare SDGs, be EXHAUSTIVE. List every keyword, pattern, and contextual cue. These SDGs have very few training examples so the classifier desperately needs detailed guidance.

Target ~4000 words. Focus on actionable classification rules with German keywords.

### Documents with RARE SDGs (analyze these in extreme detail):

"""
        # Give generous text for rare SDG docs (but not too much to avoid 429)
        for _, row in rare_docs.iterrows():
            doc_id = row['doc_id']
            txt = row['text'] if 'text' in row else ''
            active_labels = [f"SDG{i}" for i in range(1, 18) if int(row[f"SDG{i}"]) == 1]
            label_str = ', '.join(active_labels) if active_labels else 'NONE'
            truncated_txt = txt[:5000] + "...(truncated)" if len(txt) > 5000 else txt
            prompt += f"--- {doc_id} | Labels: {label_str} ---\n{truncated_txt}\n\n"

        prompt += "\n### Other Supplementary Documents:\n\n"
        for _, row in non_rare_docs.iterrows():
            doc_id = row['doc_id']
            txt = row['text'] if 'text' in row else ''
            active_labels = [f"SDG{i}" for i in range(1, 18) if int(row[f"SDG{i}"]) == 1]
            label_str = ', '.join(active_labels) if active_labels else 'NONE (zero-label)'
            truncated_txt = txt[:2000] + "...(truncated)" if len(txt) > 2000 else txt
            prompt += f"--- {doc_id} | Labels: {label_str} ---\n{truncated_txt}\n\n"

        prompt += "\nOutput ONLY the comprehensive classification guide in markdown format. No preamble."

        # Retry with exponential backoff for rate limiting
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    wait_time = 60 * (2 ** (attempt - 1))  # 60s, 120s
                    print(f"  [Analyst/Retry {attempt}/{max_retries}] Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)

                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(
                            thinking_level=types.ThinkingLevel.MEDIUM
                        ),
                        temperature=0.3,
                    )
                )
                ruleset = response.text
                with open(cache_file, "w", encoding="utf-8") as rf:
                    rf.write(ruleset)
                print(f"[Analyst] Generated and saved comprehensive ruleset to {cache_file} ({len(ruleset)} chars)")
                return ruleset
            except Exception as e:
                print(f"  [Analyst/Attempt {attempt+1}/{max_retries}] Failed: {e}")
                if attempt == max_retries - 1:
                    print("[Analyst] All retries exhausted. Using fallback ruleset.")
                    ruleset = "Fallback: Use the reference library examples and SDG definitions for classification."
                    return ruleset

        return ruleset

    # ── Threshold Tuning ─────────────────────────────────────

    def tune_thresholds(self, vote_avg_list, true_labels):
        """
        Find optimal per-SDG thresholds that maximize macro F1.
        vote_avg_list: list of dicts with vote averages per SDG.
        true_labels: list of lists with true binary labels.
        """
        y_true = np.array(true_labels)
        best_thresholds = {}
        candidate_thresholds = [0.0, 1/3 - 0.01, 1/3 + 0.01, 0.5, 2/3 - 0.01, 2/3 + 0.01, 1.0]

        for j in range(17):
            sdg = f"SDG{j+1}"
            avgs = np.array([v[sdg] for v in vote_avg_list])
            best_f1 = -1
            best_t = 0.5
            for t in candidate_thresholds:
                preds = (avgs >= t).astype(int)
                f1 = f1_score(y_true[:, j], preds, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = t
            best_thresholds[sdg] = best_t

        return best_thresholds

    # ── Full Analyst Pipeline ────────────────────────────────

    def run(self, labels_df, ruleset_cache_file="micro_ruleset.md"):
        """
        Execute the full Analyst pipeline:
          1. Select Reference Library
          2. Generate Classification Ruleset
          3. Return knowledge artifacts for the Classifier Agent
        """
        print("\n" + "=" * 65)
        print("  AGENT 1 — The Analyst Agent (The Ruleset Generator)")
        print("=" * 65)

        print("\n[Analyst] Selecting reference library from training documents...")
        ref_docs, remaining_df = self.select_reference_library(labels_df)
        ref_ids = [d['doc_id'] for d in ref_docs]

        print("[Analyst] Generating classification ruleset from remaining documents...")
        micro_ruleset = self.generate_ruleset(remaining_df, ref_ids, ruleset_cache_file)

        print("[Analyst] Knowledge artifacts ready ✓")
        return ref_docs, micro_ruleset

    def run_validation(self, labels_df):
        """
        Run the Analyst in 2-fold cross-validation mode.
        Returns two sets of knowledge artifacts (one per fold).
        """
        print("\n" + "=" * 65)
        print("  AGENT 1 — The Analyst Agent (Validation Mode, 2 Folds)")
        print("=" * 65)

        split_1_df = labels_df.iloc[:50].reset_index(drop=True)
        split_2_df = labels_df.iloc[50:].reset_index(drop=True)

        # Fold 1: Train on split_1, produce knowledge for classifying split_2
        print("\n[Analyst] === FOLD 1: Analyzing docs 0-49 ===")
        ref_docs_1, remaining_1 = self.select_reference_library(split_1_df)
        ref_ids_1 = [d['doc_id'] for d in ref_docs_1]
        ruleset_1 = self.generate_ruleset(remaining_1, ref_ids_1, "micro_ruleset_val_fold_1.md")

        # Fold 2: Train on split_2, produce knowledge for classifying split_1
        print("\n[Analyst] === FOLD 2: Analyzing docs 50-99 ===")
        ref_docs_2, remaining_2 = self.select_reference_library(split_2_df)
        ref_ids_2 = [d['doc_id'] for d in ref_docs_2]
        ruleset_2 = self.generate_ruleset(remaining_2, ref_ids_2, "micro_ruleset_val_fold_2.md")

        return (ref_docs_1, ruleset_1, split_2_df), (ref_docs_2, ruleset_2, split_1_df)


# ═════════════════════════════════════════════════════════════
# AGENT 2 — The Classifier Agent (The Forecaster)
# ═════════════════════════════════════════════════════════════
#
# Responsibilities:
#   • Build the classification prompt from Analyst-provided knowledge
#   • Cache the static prompt prefix for efficient inference
#   • Run N self-consistency voting rounds per document
#   • Aggregate votes via majority voting + threshold application
#   • Produce final binary predictions (CSV)
#
# The Classifier Agent is purely an inference engine.  It does NOT
# learn from training data directly — all knowledge is received
# from the Analyst Agent.
# ═════════════════════════════════════════════════════════════

class ClassifierAgent:
    """
    The Classifier Agent (The Forecaster).

    Receives knowledge artifacts from the Analyst Agent and classifies
    documents using self-consistency majority voting with per-SDG thresholds.
    """

    def __init__(self, client, model_id):
        self.client = client
        self.model_id = model_id

    # ── Prompt Construction ──────────────────────────────────

    def _build_static_prefix(self, ref_docs, micro_ruleset):
        """
        Build the STATIC prefix of the prompt (parts 1-3) that is identical
        across all document predictions.  This is cached server-side to
        reduce cost.

        1. System instructions + SDG definitions + rare SDG coaching
        2. Full-text reference library (up to 25 labeled docs)
        3. Comprehensive ruleset (disambiguation + rare SDG guide)
        """
        sys_section = f"""You are an expert classifier for UN Sustainable Development Goals (SDGs) in German political documents.

## SDG Definitions
{SDGS_DEFS}

## Output Format
Output a strictly valid JSON object (no markdown codeblocks). Keys: "SDG1" through "SDG17". Values: 1 if addressed, 0 if not.
Example: {{"SDG1": 0, "SDG2": 0, "SDG3": 1, "SDG4": 0, "SDG5": 0, "SDG6": 0, "SDG7": 0, "SDG8": 0, "SDG9": 0, "SDG10": 0, "SDG11": 0, "SDG12": 0, "SDG13": 0, "SDG14": 0, "SDG15": 0, "SDG16": 0, "SDG17": 0}}

## Classification Rules
- A document can have MULTIPLE SDGs (multi-label) or ZERO SDGs.
- Only label an SDG if the document SUBSTANTIVELY addresses that goal, not just a passing mention.
- When in doubt between two similar SDGs, use the disambiguation guide below.

## CRITICAL: Rare SDG Detection Guidance
The following SDGs are rare but MUST be detected when present. Do NOT ignore them:

### SDG6 (Clean Water and Sanitation)
- **Trigger keywords:** Verschlickung, Sediment, Elbe, Wassertiefe, Baggergut, Fahrrinne, Tideelbe, Gewässer, Wasserschutz, Klärschlamm, Abwasser, Trinkwasser, Wasserqualität, Wasserversorgung, Deich, Hochwasser, Wasserrahmenrichtlinie, Wasserhaushalt, Entwässerung, Kanalisation, Hafenbecken, Ingenieurwasserbau, Kanal, Steendiekkanal, Wasserbau, Uferbefestigung, Spundwand, Gewässerunterhaltung
- **Context patterns:** Documents about river management, dredging, water depth maintenance, waterway management, flood protection, sewage/wastewater, drinking water quality, canal construction/maintenance, hydraulic engineering contracts
- **Often co-occurs with:** SDG14 (when marine/sea aspects involved), SDG11 (urban water infrastructure)
- **Key distinction:** SDG6 = freshwater systems, water supply, sanitation infrastructure, canal/waterway engineering. SDG14 = ocean/marine. If a document discusses BOTH river/water management AND sea dumping, label BOTH SDG6 AND SDG14. Construction contracts by 'Ingenieurwasserbau' companies working on canals/waterways should trigger SDG6.

### SDG15 (Life on Land)
- **Trigger keywords:** Trockenrasenfläche, Garten- und Landschaftsbau, Naturschutzgebiet, Baumschutz, Grünflächen, Biotop, Artenschutz, Flora, Fauna, Wald, Waldschutz, Bodenschutz, Landschaftsplanung, ökologisch, Gehölzschnitt, Begrünung, Erdarbeiten, Vegetationsfläche, Bepflanzung, Renaturierung, Ausgleichsmaßnahmen
- **Context patterns:** Construction projects with environmental compensation measures, landscaping contracts, nature conservation areas, tree protection, ecological assessments
- **Often co-occurs with:** SDG11 (urban green spaces), SDG9 (infrastructure with environmental impact)
- **Key distinction:** ANY document mentioning significant landscaping, ecological compensation, terrestrial ecosystem management, or nature protection should trigger SDG15, even in construction/infrastructure contexts.

### SDG17 (Partnerships for the Goals)
- **Trigger keywords:** Partnership, Kooperation, Zusammenarbeit, internationale Zusammenarbeit, Entwicklungshilfe, Entwicklungszusammenarbeit, Eine-Welt, Fair Trade, Fairtrade, Partnerstadt, Städtepartnerschaft, global, multilateral, bilateral, Handelsabkommen, CETA, TTIP, Freihandel, Gründerförderung, Hochschulkooperation, Startup-Ökosystem, beyourpilot, Netzwerk, Bündelung, Plattform, Partner, Forschungseinrichtungen, gemeinsam
- **Context patterns:** International cooperation agreements, trade agreements, fair trade policies, city partnerships, development cooperation, global partnership programs, university-industry partnerships, startup ecosystem collaboration platforms, multi-institutional cooperation for innovation
- **Often co-occurs with:** SDG8 (trade/economic aspects), SDG16 (institutional cooperation), SDG9 (innovation partnerships)
- **Key distinction:** SDG17 is specifically about PARTNERSHIPS and cooperation mechanisms between entities (cities, countries, organizations, universities, research institutions) for sustainable development. Documents about startup platforms that bundle resources across multiple universities/institutions, or documents about international/inter-city cooperation frameworks, trigger SDG17.
"""

        # Reference Library section
        ref_section = "\n## Reference Library: Labeled Training Documents\n"
        ref_section += "Study these carefully. Each document is shown with its correct SDG labels.\n\n"

        for doc in ref_docs:
            active = doc['active_sdgs']
            label_display = ', '.join(active) if active else 'NONE (no SDGs)'
            ref_section += f"### {doc['doc_id']} → Labels: [{label_display}]\n"
            ref_section += f'"""{doc["text"]}"""\n\n'

        # Comprehensive ruleset section
        rule_section = "\n## Classification Guide & Disambiguation Rules\n"
        rule_section += micro_ruleset
        rule_section += "\n\n"

        return sys_section + ref_section + rule_section

    def _build_target_prompt(self, document_text):
        """
        Build the DYNAMIC per-document suffix (part 4: target document).
        This is the only part that changes between API calls.
        """
        return f"""---

## Target Document (PREDICT THIS)
Analyze the following document and predict which SDGs it addresses.

IMPORTANT CHECKLIST before finalizing your answer:
1. Did you check for SDG6 triggers (water, rivers, dredging, sanitation, floods)?
2. Did you check for SDG15 triggers (landscaping, nature conservation, ecological compensation, green spaces)?
3. Did you check for SDG17 triggers (international cooperation, partnerships, trade agreements, fair trade)?
4. Did you check for SDG2 triggers (food, nutrition, agriculture, hunger)?
5. Did you check for SDG5 triggers (gender equality, women, parental leave, equality)?
6. Did you check for SDG14 triggers (ocean, marine, sea, coastal)?

Use the reference library and classification guide above.

\"\"\"{document_text}\"\"\"

Respond ONLY with the JSON dictionary."""

    # ── Context Caching ──────────────────────────────────────

    def _create_context_cache(self, ref_docs, micro_ruleset, display_name="sdg-context"):
        """
        Create a context cache from the static prefix (parts 1-3).
        This avoids re-processing the same ~100k+ tokens for every document.
        """
        static_prefix = self._build_static_prefix(ref_docs, micro_ruleset)

        print(f"[Classifier] Creating context cache '{display_name}' (~{len(static_prefix)} chars)...")

        cached_content = self.client.caches.create(
            model=self.model_id,
            config=types.CreateCachedContentConfig(
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part(text=static_prefix)]
                    )
                ],
                ttl="72000s",  # 20 hour TTL – enough for a full prediction run
                display_name=display_name,
            )
        )

        print(f"[Classifier] Context cache created: {cached_content.name}")
        print(f"  Usage: {cached_content.usage_metadata}")
        return cached_content

    # ── Single API Call ──────────────────────────────────────

    def _single_api_call(self, prompt, doc_id, vote_num, cached_content=None):
        """Make a single API call with retry logic.  Uses context cache if provided."""
        retries = 5
        while retries > 0:
            try:
                config_kwargs = dict(
                    thinking_config=types.ThinkingConfig(
                        thinking_level=types.ThinkingLevel.HIGH
                    ),
                    temperature=0.4,
                    response_mime_type="application/json",
                )
                if cached_content is not None:
                    config_kwargs["cached_content"] = cached_content.name

                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=prompt,
                    config=types.GenerateContentConfig(**config_kwargs)
                )
                return parse_json_response(response.text)
            except Exception as e:
                retries -= 1
                err_str = str(e)
                if '429' in err_str or 'RESOURCE_EXHAUSTED' in err_str:
                    wait_time = 30 * (6 - retries)
                    print(f"    [Classifier/Rate Limited] {doc_id} vote {vote_num}: waiting {wait_time}s... ({retries} retries left)")
                    time.sleep(wait_time)
                else:
                    print(f"    [Classifier/API Error] {doc_id} vote {vote_num}: {e}. Retrying in 5s... ({retries} retries left)")
                    time.sleep(5)
        return None

    # ── Vote Aggregation ─────────────────────────────────────

    @staticmethod
    def majority_vote(vote_dicts):
        """Given a list of prediction dicts, return vote averages per SDG."""
        avg_dict = {}
        for j in range(1, 18):
            key = f"SDG{j}"
            vals = [v.get(key, 0) for v in vote_dicts]
            avg_dict[key] = sum(vals) / len(vals)
        return avg_dict

    @staticmethod
    def apply_thresholds(avg_dict, thresholds=None):
        """Convert vote averages to binary using per-SDG thresholds."""
        if thresholds is None:
            thresholds = {f"SDG{j}": 0.5 for j in range(1, 18)}
        result = {}
        for j in range(1, 18):
            key = f"SDG{j}"
            result[key] = 1 if avg_dict[key] >= thresholds[key] else 0
        return result

    # ── Full Prediction Pipeline ─────────────────────────────

    def predict(self, test_df, test_dir, ref_docs, micro_ruleset,
                label_mode=False, output_filename=None, optimal_thresholds=None):
        """
        Execute the full Classifier pipeline on a set of documents.

        Args:
            test_df: DataFrame with at minimum a 'doc_id' column
            test_dir: Directory containing .txt files
            ref_docs: Reference library from the Analyst Agent
            micro_ruleset: Classification ruleset from the Analyst Agent
            label_mode: If True, also collect ground-truth labels (validation)
            output_filename: If set, checkpoint predictions to this file
            optimal_thresholds: Per-SDG thresholds from the Analyst Agent
        """
        print("\n" + "=" * 65)
        print("  AGENT 2 — The Classifier Agent (The Forecaster)")
        print("=" * 65)

        all_vote_avgs = []
        all_preds_test = []
        val_true_labels = [] if label_mode else None

        # Checkpoint loading
        if output_filename and os.path.exists(output_filename):
            try:
                checkpoint_df = pd.read_csv(output_filename)
                completed_docs = set(checkpoint_df['doc_id'].astype(str))

                for _, row in checkpoint_df.iterrows():
                    pred_dict = {'doc_id': str(row['doc_id'])}
                    avg_dict = {}
                    for i in range(1, 18):
                        pred_dict[f"SDG{i}"] = int(row[f"SDG{i}"])
                        avg_dict[f"SDG{i}"] = float(row[f"SDG{i}"])
                    all_preds_test.append(pred_dict)
                    all_vote_avgs.append(avg_dict)

                test_df = test_df[~test_df['doc_id'].astype(str).isin(completed_docs)]
                print(f"[Classifier] Loaded {len(completed_docs)} predictions from checkpoint. {len(test_df)} remaining.")
            except Exception as e:
                print(f"[Classifier] Failed to load checkpoint from {output_filename}: {e}")

        # Skip docs before RESUME_FROM_DOC
        if RESUME_FROM_DOC:
            test_df = test_df[test_df['doc_id'].apply(
                lambda x: int(str(x).replace('doc_', '')) >= RESUME_FROM_DOC
            )]
            print(f"[Classifier] Resuming from doc_{RESUME_FROM_DOC}. {len(test_df)} documents remaining.")

        # Create context cache for the static prefix
        cached_content = None
        if len(test_df) > 0:
            try:
                cached_content = self._create_context_cache(
                    ref_docs, micro_ruleset,
                    display_name=f"sdg-{output_filename or 'pred'}"
                )
            except Exception as e:
                print(f"[Classifier/Warning] Context cache creation failed: {e}")
                print("  Falling back to non-cached mode (full prompt per call).")
                cached_content = None

        print(f"\n[Classifier] Starting predictions on {len(test_df)} documents ({NUM_VOTES} votes each)...")
        if cached_content:
            print(f"  Using context cache: {cached_content.name}")

        # Load historical vote file if configured
        historical_votes = {}
        if HISTORICAL_VOTE_FILE and os.path.exists(HISTORICAL_VOTE_FILE) and not label_mode:
            hist_df = pd.read_csv(HISTORICAL_VOTE_FILE)
            for _, hrow in hist_df.iterrows():
                hid = str(hrow['doc_id'])
                historical_votes[hid] = {f"SDG{j}": int(hrow[f"SDG{j}"]) for j in range(1, 18)}
            api_votes_needed = max(1, NUM_VOTES - 1)
            print(f"  Loaded {len(historical_votes)} historical votes from {HISTORICAL_VOTE_FILE}")
            print(f"  Will run {api_votes_needed} fresh API votes + 1 historical vote = {api_votes_needed + 1} total votes")
        else:
            api_votes_needed = NUM_VOTES

        docs_processed = 0
        for i, row in test_df.iterrows():
            doc_id = row['doc_id']
            doc_path = os.path.join(test_dir, f"{doc_id}.txt")
            if os.path.exists(doc_path):
                with open(doc_path, 'r', encoding='utf-8') as f:
                    doc_text = f.read()
            else:
                if 'text' in row:
                    doc_text = row['text']
                else:
                    print(f"Warning: {doc_path} not found.")
                    doc_text = ""

            # Build prompt: if cached, send only the target doc; else send full prompt
            if cached_content:
                prompt = self._build_target_prompt(doc_text)
            else:
                prompt = self._build_static_prefix(ref_docs, micro_ruleset) + self._build_target_prompt(doc_text)

            # Collect votes: inject historical vote first, then fresh API calls
            votes = []

            doc_id_str = str(doc_id)
            if doc_id_str in historical_votes:
                votes.append(historical_votes[doc_id_str])

            fresh_needed = api_votes_needed
            for v in range(fresh_needed):
                pred = self._single_api_call(prompt, doc_id, v + 1, cached_content=cached_content)
                if pred is not None:
                    votes.append(pred)
                else:
                    print(f"    [Classifier/Failed] {doc_id} vote {v+1}. Skipping this vote.")
                time.sleep(2)

            # Aggregate votes
            if len(votes) == 0:
                print(f"  [Classifier/Failed] All votes failed for {doc_id}. Defaulting to zeros.")
                avg_dict = {f"SDG{j}": 0.0 for j in range(1, 18)}
            else:
                avg_dict = self.majority_vote(votes)

            pred_dict = self.apply_thresholds(avg_dict)
            pred_dict['doc_id'] = doc_id
            all_preds_test.append(pred_dict)
            all_vote_avgs.append(avg_dict)

            if label_mode:
                val_true_labels.append([row[f"SDG{j}"] for j in range(1, 18)])

            docs_processed += 1
            if docs_processed % 5 == 0 or docs_processed == 1:
                print(f"  [Classifier] [{docs_processed}/{len(test_df)}] {doc_id}: {len(votes)}/{NUM_VOTES} votes OK")

            # Save checkpoint every 50 docs
            if output_filename and docs_processed % 50 == 0:
                self._save_checkpoint(all_preds_test, all_vote_avgs, output_filename, optimal_thresholds, label_mode)

        # Clean up cache
        if cached_content:
            try:
                self.client.caches.delete(name=cached_content.name)
                print(f"  [Classifier/Cache] Deleted context cache: {cached_content.name}")
            except Exception as e:
                print(f"  [Classifier/Cache] Failed to delete cache (will expire via TTL): {e}")

        if label_mode:
            return all_preds_test, all_vote_avgs, val_true_labels
        return all_preds_test, all_vote_avgs

    def _save_checkpoint(self, all_preds_test, all_vote_avgs, output_filename, optimal_thresholds, label_mode):
        """Save prediction checkpoints to CSV + Google Drive."""
        columns = ['doc_id'] + SDG_COLS
        checkpoint_out = pd.DataFrame(all_preds_test)
        for col in columns:
            if col not in checkpoint_out.columns:
                checkpoint_out[col] = 0
        checkpoint_out = checkpoint_out[columns]
        for col in columns[1:]:
            checkpoint_out[col] = checkpoint_out[col].astype(int).clip(0, 1)
        checkpoint_out.to_csv(output_filename, index=False)
        print(f"  [Classifier/Checkpoint] Saved majority to {output_filename} ({len(all_preds_test)} docs)")
        save_to_drive(output_filename)

        # Save tuned checkpoint if thresholds available
        if optimal_thresholds and not label_mode:
            tuned_preds = []
            for k, avg in enumerate(all_vote_avgs):
                tp = self.apply_thresholds(avg, optimal_thresholds)
                tp['doc_id'] = all_preds_test[k]['doc_id']
                tuned_preds.append(tp)
            tuned_out = pd.DataFrame(tuned_preds)
            for col in columns:
                if col not in tuned_out.columns:
                    tuned_out[col] = 0
            tuned_out = tuned_out[columns]
            for col in columns[1:]:
                tuned_out[col] = tuned_out[col].astype(int).clip(0, 1)
            tuned_out.to_csv("submission_tuned.csv", index=False)
            print(f"  [Classifier/Checkpoint] Saved tuned to submission_tuned.csv ({len(tuned_preds)} docs)")
            save_to_drive("submission_tuned.csv")


# ═════════════════════════════════════════════════════════════
# Orchestrator — Coordinates both agents
# ═════════════════════════════════════════════════════════════

def generate_report(analyst, preds_1, avgs_1, true_1, preds_2, avgs_2, true_2):
    """Generate validation report with majority vote AND threshold-tuned metrics."""
    # === MAJORITY VOTE METRICS ===
    y_true_1 = np.array(true_1)
    y_pred_1 = np.array([[p.get(f"SDG{j}", 0) for j in range(1, 18)] for p in preds_1])
    score_1, p_1, r_1, per_sdg_f1_1, per_sdg_p_1, per_sdg_r_1 = evaluate_metrics(y_true_1, y_pred_1)

    y_true_2 = np.array(true_2)
    y_pred_2 = np.array([[p.get(f"SDG{j}", 0) for j in range(1, 18)] for p in preds_2])
    score_2, p_2, r_2, per_sdg_f1_2, per_sdg_p_2, per_sdg_r_2 = evaluate_metrics(y_true_2, y_pred_2)

    all_preds = preds_1 + preds_2
    all_avgs = avgs_1 + avgs_2
    all_trues = true_1 + true_2
    y_true_val = np.array(all_trues)
    y_pred_val = np.array([[p.get(f"SDG{j}", 0) for j in range(1, 18)] for p in all_preds])
    score_all, p_all, r_all, per_sdg_f1, per_sdg_p, per_sdg_r = evaluate_metrics(y_true_val, y_pred_val)

    # === THRESHOLD TUNING (Analyst Agent) ===
    print("\n[Analyst] Tuning per-SDG thresholds on combined validation data...")
    optimal_thresholds = analyst.tune_thresholds(all_avgs, all_trues)

    # Apply tuned thresholds
    tuned_preds = [ClassifierAgent.apply_thresholds(avg, optimal_thresholds) for avg in all_avgs]
    y_pred_tuned = np.array([[p.get(f"SDG{j}", 0) for j in range(1, 18)] for p in tuned_preds])
    score_tuned, p_tuned, r_tuned, per_sdg_f1_t, per_sdg_p_t, per_sdg_r_t = evaluate_metrics(y_true_val, y_pred_tuned)

    # === BUILD REPORT ===
    md = f"# Validation Results ({NUM_VOTES}-Vote Self-Consistency)\n\n"

    md += f"## Split 1 (Majority Vote)\n"
    md += f"**Macro F1:** {score_1:.4f} | **Precision:** {p_1:.4f} | **Recall:** {r_1:.4f}\n\n"
    md += "| SDG | F1 Score | Precision | Recall |\n|---|---|---|---|\n"
    for j in range(17):
        md += f"| SDG{j+1} | {per_sdg_f1_1[j]:.4f} | {per_sdg_p_1[j]:.4f} | {per_sdg_r_1[j]:.4f} |\n"

    md += f"\n## Split 2 (Majority Vote)\n"
    md += f"**Macro F1:** {score_2:.4f} | **Precision:** {p_2:.4f} | **Recall:** {r_2:.4f}\n\n"
    md += "| SDG | F1 Score | Precision | Recall |\n|---|---|---|---|\n"
    for j in range(17):
        md += f"| SDG{j+1} | {per_sdg_f1_2[j]:.4f} | {per_sdg_p_2[j]:.4f} | {per_sdg_r_2[j]:.4f} |\n"

    md += f"\n## Overall Combined (Majority Vote, threshold=0.5)\n"
    md += f"**Macro F1:** {score_all:.4f} | **Precision:** {p_all:.4f} | **Recall:** {r_all:.4f}\n\n"
    md += "| SDG | F1 Score | Precision | Recall |\n|---|---|---|---|\n"
    for j in range(17):
        md += f"| SDG{j+1} | {per_sdg_f1[j]:.4f} | {per_sdg_p[j]:.4f} | {per_sdg_r[j]:.4f} |\n"

    md += f"\n## Overall Combined (Threshold-Tuned)\n"
    md += f"**Macro F1:** {score_tuned:.4f} | **Precision:** {p_tuned:.4f} | **Recall:** {r_tuned:.4f}\n\n"
    md += "| SDG | F1 Score | Precision | Recall | Threshold |\n|---|---|---|---|---|\n"
    for j in range(17):
        sdg = f"SDG{j+1}"
        md += f"| {sdg} | {per_sdg_f1_t[j]:.4f} | {per_sdg_p_t[j]:.4f} | {per_sdg_r_t[j]:.4f} | {optimal_thresholds[sdg]:.3f} |\n"

    # Save thresholds for production use
    with open("optimal_thresholds.json", "w") as f:
        json.dump(optimal_thresholds, f, indent=2)
    print(f"[Analyst] Saved optimal thresholds to optimal_thresholds.json")

    return md, all_preds, all_trues, optimal_thresholds


# ─────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────

def main():
    ensure_test_folder()

    train_dir = "train"
    labels_file = "train_labels.csv"

    if not os.path.exists(labels_file):
        print(f"Error: {labels_file} not found.")
        sys.exit(1)

    labels_df = pd.read_csv(labels_file)

    # Load all training document texts
    docs = []
    print("Loading training text documents...")
    for doc_id in labels_df['doc_id']:
        doc_path = os.path.join(train_dir, f"{doc_id}.txt")
        if os.path.exists(doc_path):
            with open(doc_path, 'r', encoding='utf-8') as f:
                docs.append(f.read())
        else:
            print(f"Warning: {doc_path} not found.")
            docs.append("")
    labels_df['text'] = docs

    client = get_gemini_client()
    model_id = "gemini-3.1-pro-preview"

    # Instantiate the two agents
    analyst = AnalystAgent(client, model_id, train_dir)
    classifier = ClassifierAgent(client, model_id)

    if VALIDATION_MODE:
        print(f"\n--- RUNNING IN VALIDATION MODE (2 Splits, {NUM_VOTES}-Vote Self-Consistency) ---")

        # ── Agent 1: Analyst produces knowledge for both folds ──
        fold_1, fold_2 = analyst.run_validation(labels_df)
        ref_docs_1, ruleset_1, test_df_1 = fold_1
        ref_docs_2, ruleset_2, test_df_2 = fold_2

        # ── Agent 2: Classifier predicts on both folds ──
        print("\n[Classifier] === FOLD 1: Classifying docs 50-99 ===")
        preds_1, avgs_1, true_1 = classifier.predict(
            test_df_1, train_dir,
            ref_docs_1, ruleset_1,
            label_mode=True
        )

        print("\n[Classifier] === FOLD 2: Classifying docs 0-49 ===")
        preds_2, avgs_2, true_2 = classifier.predict(
            test_df_2, train_dir,
            ref_docs_2, ruleset_2,
            label_mode=True
        )

        # ── Generate and save report ──
        md_report, all_preds, all_trues, optimal_thresholds = generate_report(
            analyst, preds_1, avgs_1, true_1, preds_2, avgs_2, true_2
        )

        with open("validation_results.md", "w", encoding="utf-8") as f:
            f.write(md_report)
        print(md_report)

        # Save validation predictions
        columns = ['doc_id'] + SDG_COLS
        submission_df = pd.DataFrame(all_preds)
        for col in columns:
            if col not in submission_df.columns:
                submission_df[col] = 0
        submission_df = submission_df[columns]
        for col in columns[1:]:
            submission_df[col] = submission_df[col].astype(int).clip(0, 1)
        submission_df.to_csv("val_predictions.csv", index=False)
        print(f"Saved predictions to val_predictions.csv")

    else:
        print("\n--- RUNNING IN FULL PREDICTION MODE ---")

        # ── Agent 1: Analyst produces knowledge from all training data ──
        ref_docs, micro_ruleset = analyst.run(labels_df)

        test_skeleton_file = "test_skeleton.csv"
        if not os.path.exists(test_skeleton_file):
            print(f"Error: {test_skeleton_file} not found.")
            sys.exit(1)

        test_df = pd.read_csv(test_skeleton_file)
        test_dir = "test"
        output_filename = "submission_majority.csv"

        # Load thresholds (produced by earlier Analyst validation run)
        loaded_thresholds = None
        if os.path.exists("optimal_thresholds.json"):
            with open("optimal_thresholds.json", "r") as f:
                loaded_thresholds = json.load(f)
            print(f"[Analyst] Loaded tuned thresholds from optimal_thresholds.json")

        # ── Agent 2: Classifier predicts on test set ──
        all_preds, all_avgs = classifier.predict(
            test_df, test_dir,
            ref_docs, micro_ruleset,
            label_mode=False, output_filename=output_filename,
            optimal_thresholds=loaded_thresholds
        )

        # Save majority vote submission (safe baseline)
        majority_df = pd.DataFrame(all_preds)
        for col in SDG_COLS:
            if col not in majority_df.columns:
                majority_df[col] = 0
        majority_df = majority_df[['doc_id'] + SDG_COLS]
        for col in SDG_COLS:
            majority_df[col] = majority_df[col].astype(int).clip(0, 1)
        majority_df.to_csv("submission_majority.csv", index=False)
        print(f"Saved majority-vote submission to submission_majority.csv")
        save_to_drive("submission_majority.csv")

        # Save threshold-tuned submission (if thresholds available)
        if loaded_thresholds:
            print(f"Applying tuned thresholds for final submission_tuned.csv")
            tuned_preds = [ClassifierAgent.apply_thresholds(avg, loaded_thresholds) for avg in all_avgs]
            for i, p in enumerate(tuned_preds):
                p['doc_id'] = all_preds[i]['doc_id']
            tuned_df = pd.DataFrame(tuned_preds)
            for col in SDG_COLS:
                if col not in tuned_df.columns:
                    tuned_df[col] = 0
            tuned_df = tuned_df[['doc_id'] + SDG_COLS]
            for col in SDG_COLS:
                tuned_df[col] = tuned_df[col].astype(int).clip(0, 1)
            tuned_df.to_csv("submission_tuned.csv", index=False)
            print(f"Saved threshold-tuned submission to submission_tuned.csv")
            save_to_drive("submission_tuned.csv")

            # Show diff summary
            diff_count = (majority_df[SDG_COLS].values != tuned_df[SDG_COLS].values).sum()
            print(f"  Differences between majority vs tuned: {diff_count} label changes across {len(majority_df)} documents")
        else:
            print("No optimal_thresholds.json found. Only majority-vote submission saved.")


if __name__ == "__main__":
    main()
