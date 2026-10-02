# Amazon ML Challenge 2026: Business Entity Resolution

> **Final Leaderboard Score:** **`0.98447`** (Submission S008)  
> **Task:** High-precision Entity Matching across multi-source commercial business records  
> **Key Architecture:** Hybrid Retrieval (BM25 + Fine-Tuned Dense Bi-Encoder) &rarr; Feature Engineering + Cross-Encoder Rerankers &rarr; Stage-2 LightGBM &rarr; Max-Claimer Conflict Resolution  

---

## Project Overview

Imagine searching for a local shop across multiple online directories:
- Directory A calls it: `"Joe's Pizza Inc., 123 Main St, New York"`
- Directory B calls it: `"Joes Pizzeria, 123 Main Street Suite 1, NY"`
- Directory C calls it: `"Joe Pizza, Main Rd, New York"` (with a typo and missing details)

Even though all three refer to the exact same real-world business, they share **no common ID numbers**, have inconsistent spellings, abbreviations, or missing information.

**Business Entity Resolution** is the machine learning task of matching these fragmented, noisy records across disparate databases back to their single canonical reference profile.

---

## The Dataset Explained

The competition dataset mirrors real-world enterprise entity resolution at massive scale (millions of records):

### 1. Data Sources
Each record belongs to one of three sources (indicated by its ID prefix):
* **Source 1 (`S1-*`):** The clean, deduplicated **ground truth reference** (the master directory).
* **Source 2 (`S2-*`):** A noisy secondary source containing real-world variations, abbreviations, and typos.
* **Source 3 (`S3-*`):** An even noisier tertiary source with severe address corruptions and missing fields.

### 2. Available Fields
Every file is a Tab-Separated Values (`.tsv`) file with four core fields:
* `entity_id`: Unique identifier (e.g., `S1-00042`, `S2-01934`, `S3-08123`).
* `business_name`: Company/shop name (subject to typos, abbreviations, brand changes, or regional scripts).
* `business_address`: Physical address (often missing components, reordered words, or landmark descriptions like *"Near SBI ATM"*).
* `country`: Geographic territory (`US`, `India`, and in the test set, `France`).

### 3. Dataset Scale
| Split | Source 1 (`S1`) | Source 2 (`S2`) | Source 3 (`S3`) | Countries Covered | Ground Truth Links |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | ~2.21 Million | ~5.03 Million | ~5.29 Million | US (60%), India (40%) | 7.64 Million pairs |
| **Test** | ~1.73 Million | ~4.89 Million | ~5.08 Million | India (46.8%), US (38.3%), **France (15.0%)** | Hidden (evaluated on portal) |

### 4. Key Data Quirks & Challenges
* **The "France" Surprise (Out-of-Distribution):** The training set contains *only* US and India data. However, 15% of the test set (~259,452 reference businesses) is from **France**, where models had **zero training labels** and had to generalize completely zero-shot.
* **Severe Noise & Transliteration:** ~23% of raw Indian records in Source 2 were written in regional Indic scripts (Devanagari, Tamil, Telugu, Kannada, etc.) while Source 1 was always in English/Latin characters.
* **Singletons & Decoys:** ~5.6% of Source 1 businesses have **no matches at all** (singletons), and test sets contain millions of deceptive "decoy" records designed to trick models into false positives.
* **Exact Matching Fails:** ~78% of true matches do not have an exact name match; exact name + address matching covers less than 3% of the dataset.

### 5. Evaluation Metric: Macro $F_{0.5}$
The competition is evaluated on **Macro $F_{0.5}$ score** averaged across all Source 1 entities:
$$\text{Macro } F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

> **Why $F_{0.5}$ matters:** It penalizes False Positives **twice as heavily** as False Negatives. In business entity resolution, wrongly linking two different businesses is considered twice as damaging as failing to link one.

---

## Our Experiments: The Journey from 0.77 to 0.984+

We tackled the problem in systematic, hypothesis-driven phases:

```
[Millions of Records]
        │
        ▼
1. FAST CANDIDATE BLOCKING (99.8% Recall)
   ├── Character 4-gram BM25 Lexical Search (Name + Address & Name-only)
   └── Fine-Tuned Multilingual-E5-Small Dense Bi-Encoder
        │
        ▼
2. BASE LIGHTGBM FILTER
   └── 22 Core Lexical/Address Features ➔ Ranks top-10 candidates per S1
        │
        ▼
3. DEEP CROSS-ENCODER RERANKING
   └── Multilingual-E5-Large (560M params) Joint Text Scoring
        │
        ▼
4. STAGE-2 LIGHTGBM (118 Engineered Features)
   ├── Numeric & House-Number Consistency Blocks
   ├── Relational & Competing-Owner Signals
   └── Token-Difference & Address Concordance
        │
        ▼
5. POST-PROCESSING & CONFLICT RESOLUTION
   ├── Max-Claimer (Enforce 1-to-1 Observation Exclusivity)
   └── France Shift Decoy Removal Patch
        │
        ▼
   [Final Score: 0.98447]
```

### Phase 1: Exploration & The Retrieval Foundation (RECON)
* **The Problem:** Comparing every Source 1 business against 10 million candidate records directly is computationally impossible ($1.7\text{M} \times 10\text{M} \approx 17\text{ trillion}$ pairs).
* **The Solution (Candidate Blocking):** We built a high-speed lexical search engine using character 4-gram BM25. Searching name + address combined with a name-only fallback captured **94.5%** of all true matches while filtering out 99.99% of non-matching records.
* **First Models:** Switching from baseline Logistic Regression (77.59% validation) to LightGBM jumped performance immediately to **87.53%**, proving that non-linear decision trees were needed to balance fuzzy name and address signals.

### Phase 2: Feature Engineering & Domain Intelligence (E008–E014)
* **Relational Context:** A business isn't matched in a vacuum. We engineered competition features: how much higher did candidate #1 score compared to candidate #2? If one candidate is an overwhelming match, other weak candidates for that business are rejected.
* **House Number & Numeric Auditing:** Deceptive synthetic decoys often had the exact same business name but modified house numbers (e.g., `#104` instead of `#102`). We added 27 numeric-address conflict features, preventing thousands of false links and lifting validation to **94.02%**.
* **Indic Transliteration:** We built a custom transliteration module converting Devanagari, Tamil, Telugu, and other Indic scripts into Latin phonetics, recovering hundreds of missed Indian businesses (+1.2 pp boost on India).

### Phase 3: Dense Semantic Retrieval & Cross-Encoders (E018–E024)
* **Fine-Tuned Bi-Encoder:** To catch matches with completely different wordings that lexical BM25 missed, we fine-tuned a `multilingual-e5-small` bi-encoder using InfoNCE contrastive learning on 400,000 entities. Merging dense retrieval with BM25 raised our candidate recall from 94.5% to **99.79%**!
* **Cross-Encoder Rerankers:** Fast retrieval gave us ~128 candidates per entity. We used LightGBM to select the top-10 most promising candidates and fed them into deep multilingual cross-encoders (`multilingual-e5-base` and later `multilingual-e5-large`), allowing the model to read full text pairs jointly. This produced our first major leaderboard leap to **0.97977** (S004).

### Phase 4: Exclusivity & "The Max-Claimer" (E020, RL-27)
* **The Insight:** In our data, any single Source 2 or Source 3 record can belong to **at most one** Source 1 business entity in reality. If multiple Source 1 businesses try to claim the same record, at least one of them must be wrong!
* **The Algorithm:** We implemented the **Max-Claimer**: when a record is accepted by multiple businesses, it is awarded exclusively to the business with the highest prediction probability.
* **Competing-Owner Features:** We added 10 features measuring whether other entities were competing for the same record. This directly eliminated hundreds of ambiguous false positives, driving our score to **0.98198** (S005).

### Phase 5: Scaling to E5-Large & The French Shift Discovery (E026–E027 &rarr; Final S008)
* **Scaling to 560M Parameters (`rrL` / `rrL2`):** We trained a `multilingual-e5-large` cross-encoder specifically focused on hard edge cases (businesses sharing identical addresses, differing by a single token, or missing address strings). This pushed validation to **99.03%** and leaderboard to **0.98371** (S006).
* **Cracking the France Generalization Gap:** Because France had no training labels, synthetic generator decoys in French data were tricking models into false positives. Through mirror-census analysis, we identified that generator decoys systematically shifted house numbers by fixed offsets ($+1, +2, +3, +5, +7, +11\dots$). Pruning these guaranteed decoy links delivered our final winning submission: **`0.98447`** (S008).

---

## Leaderboard Progression Summary

| Submission | Core Innovations & Model Architecture | Macro $F_{0.5}$ (Leaderboard) | Key Milestone |
| :--- | :--- | :--- | :--- |
| **S002** | BM25 retrieval + Feature Engineering + E5-Small Reranker | `0.950650` | Strong initial baseline |
| **S004** | Hybrid Retrieval (Dense Bi-Encoder ∪ BM25) + E5-Base Reranker + Max-Claimer | `0.979772` | Jumped retrieval recall to 99.8% |
| **S005** | Added 10 Competing-Owner & Record-Centric Conflict Features | `0.981984` | Solved cross-entity collisions |
| **S006** | Scaled to `multilingual-e5-large` (560M) Cross-Encoder (113 features) | `0.983713` | Deep semantic disambiguation |
| **S008 (Final)**| Continued `rrL2` Hard-Pair Training + Diff-Token Features + France Decoy Pruning | **`0.984470`** | **Final Best Submission** |

---

## Key Takeaways & What Didn't Work

### What Worked Best:
1. **Hybrid Retrieval is King:** Lexical search is fast and captures exact words; dense embeddings capture synonyms and typos. Combining both achieved 99.8% recall.
2. **Precision-First Thinking:** Because $F_{0.5}$ rewards precision, discarding low-confidence or contested matches via the Max-Claimer provided massive gains.
3. **Structured Numeric Comparison:** Deep neural networks often gloss over tiny single-digit changes in house numbers; explicit numeric diff features stopped them from falling for decoy addresses.

### What Failed (and Why):
* ❌ **Exact-Match Blocking:** Missed ~78% of valid pairs due to spelling variations and formatting differences.
* ❌ **Naive Threshold Tuning on Sparse Validation:** Rules that worked on small validation samples failed on the full dataset because collisions only appear when data is dense.
* ❌ **Unconstrained Generative Negatives:** Generating generic synthetic negatives without country-specific context degraded validation performance on real data.

---

## Repository Structure

* [`MASTER_KNOWLEDGE.md`](MASTER_KNOWLEDGE.md) — Comprehensive technical documentation, complete experiment logs, hyperparameter tables, error budgets, and post-competition analyses.
* [`src/`](src/) — Core pipeline modules:
  * [`src/retrieval_engine.py`](src/retrieval_engine.py) — Numba-accelerated BM25 lexical candidate generator.
  * [`src/e022_biencoder.py`](src/e022_biencoder.py) — Multilingual-E5 bi-encoder dense candidate retrieval.
  * [`src/e009_numeric_features.py`](src/e009_numeric_features.py) — Address and house-number conflict analysis.
  * [`src/translit.py`](src/translit.py) — Indic script transliteration engine.
  * [`src/e023_stage2.py`](src/e023_stage2.py) — Stage-2 LightGBM model pipeline.
  * [`src/e020_maxclaimer.py`](src/e020_maxclaimer.py) — Exclusivity and conflict-resolution engine.
* [`experiments/`](experiments/) — Reproducible training scripts, evaluation notebooks, and submission artifacts for each milestone.
* [`student_resource/`](student_resource/) — Official competition statement, documentation templates, and submission validation tools.
