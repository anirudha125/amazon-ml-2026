# Amazon ML Challenge 2026 — External Download & Resource Log

This log is the authoritative record of all external tools, models, packages, and datasets considered, downloaded, or utilized in this workspace, in strict compliance with the **Amazon ML Challenge 2026 Academic Integrity and Fair Play Rules**.

---

## 1. Compliance Policy & Strict Rules

### 1.1 Strictly Prohibited Actions (Disqualification Offenses)
- **External Business Entity Lookups:** Commercial APIs (Google Places, Dun & Bradstreet, OpenCorporates, ZoomInfo, etc.).
- **Government Registry Lookups:** Company registry scraping or lookups (e.g. Ministry of Corporate Affairs India, US SEC EDGAR, French INSEE / SIRENE).
- **Geocoding APIs:** External geocoders (Google Maps, OpenStreetMap Nominatim, Mapbox, etc.) used to normalize, enrich, or resolve addresses.
- **External Data Augmentation:** Any internet scraping or external data source providing business names, addresses, or phone numbers.

### 1.2 Permitted External Resources
- Standard open-source tooling, libraries, and frameworks (e.g., PyTorch, LightGBM, RapidFuzz, Scikit-learn).
- Pretrained language / embedding models provided they satisfy:
  - **Open Source License:** MIT or Apache 2.0 license strictly.
  - **Parameter Constraint:** Up to 8 Billion parameters maximum.
  - **Zero Business Entity Leakage:** Model must NOT contain proprietary entity resolution lookup databases or cached entity graph mappings.

---

## 2. Resource Audit & Download Ledger

| Timestamp (UTC/IST) | Experiment | Resource / Package | Version | Source / Domain | License | Purpose / Reason | Business Data Included? | Compliance Assessment |
|---|---|---|---|---|---|---|---|---|
| Initial Environment Setup | E000 | `RapidFuzz` | 3.14.6 | PyPI (`pypi.org`) | MIT | High-performance C++ string similarity metrics (Levenshtein, Jaro-Winkler, Token Sort/Set) | No (Pure algorithmic library) | **COMPLIANT** |
| Initial Environment Setup | E000 | `LightGBM` | 4.7.0 | PyPI (`pypi.org`) | MIT | Gradient boosted decision trees for pairwise and relational candidate scoring | No (Pure ML framework) | **COMPLIANT** |
| Initial Environment Setup | E000 | `scikit-learn` | 1.8.0 | PyPI (`pypi.org`) | BSD-3-Clause | KFold splitting, StandardScaler, logistic regression baselines | No (Pure ML framework) | **COMPLIANT** |
| Initial Environment Setup | E000 | `psutil` | 7.0.0 | PyPI (`pypi.org`) | BSD-3-Clause | Memory and CPU resource telemetry | No (System utility) | **COMPLIANT** |
| Initial Environment Setup | E000 | `rank-bm25` | 0.2.2 | PyPI (`pypi.org`) | Apache 2.0 | Reference BM25 implementation (custom streaming BM25 implemented in `src/`) | No (Algorithmic library) | **COMPLIANT** |
| 2026-09-26 | Retrieval engine (E013 prep) | `numba` (+ dependency `llvmlite` 0.49.0) | 0.67.0 | PyPI (`pypi.org`) | BSD-2-Clause (llvmlite: BSD-2-Clause + Apache-2.0 LLVM exception) | JIT-compiled parallel BM25 posting-list scoring for scalable candidate retrieval (train expansion + test pipeline). Dry-run confirmed numpy 2.2.6 unchanged. | No (pure compiler library) | **COMPLIANT** |
| 2026-09-26 | E017 (Kaggle 2xT4, frozen-embedding diagnostic) | `intfloat/multilingual-e5-small` (Hugging Face Hub) | commit `614241f622f53c4eeff9890bdc4f31cfecc418b3` | huggingface.co/intfloat/multilingual-e5-small | MIT (model card) | ~118M-parameter multilingual sentence encoder (111 languages incl. hi/bn/ta/te/kn/ml/gu/mr/pa/or/fr); frozen cosine-similarity features for name / name+address / address. Downloaded inside the Kaggle session only. Param count confirmed at load (see experiments/E017_embed/run_info.json). | No (general-purpose language model; no entity/business lookup tables) | **COMPLIANT** (MIT, <<8B) |
| 2026-09-26 | E018b (Kaggle 2xT4, cross-encoder reranker variant) | `intfloat/multilingual-e5-base` (Hugging Face Hub) | commit `d128750597153bb5987e10b1c3493a34e5a4502a` | huggingface.co/intfloat/multilingual-e5-base | MIT (model card) | ~278M-parameter multilingual encoder (XLM-R base), fine-tuned as a pairwise match reranker on competition TRAIN pairs only (OOF). Downloaded inside the Kaggle session only. | No | **COMPLIANT** (MIT, <<8B) |
| 2026-09-26 | E018 (Kaggle 2xT4, cross-encoder reranker) | `intfloat/multilingual-e5-small` | commit `614241f622f53c4eeff9890bdc4f31cfecc418b3` | as above | MIT | Same model as E017, fine-tuned (1 epoch, OOF by S1 fold) as a pairwise reranker on competition TRAIN pairs only; 117,653,760 params. | No | **COMPLIANT** |
| 2026-09-26 18:04 UTC | A100 profile (tools/a100_profile.py) + planned E023 reranker arms (Lightning A100) | `intfloat/multilingual-e5-base` | commit `d128750597153bb5987e10b1c3493a34e5a4502a` | huggingface.co/intfloat/multilingual-e5-base | MIT (model card) | Throughput/VRAM benchmark; candidate larger reranker / bi-encoder backbone (fine-tuned on competition TRAIN pairs only). ~278M params. | No | **COMPLIANT** (MIT, <<8B) |
| 2026-09-26 19:20 UTC | A100 session (debugging a stalled LightGBM job) | `py-spy` | latest (PyPI) | PyPI (`pypi.org`) | MIT | Sampling profiler used once to dump a Python stack; development tool only, not part of the pipeline | No | **COMPLIANT** |
| 2026-09-27 05:11 UTC | E026_rrL (Lightning A100-SXM4-40GB, reranker stream / Session B) | `intfloat/multilingual-e5-large` (Hugging Face Hub) | commit `3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3` | huggingface.co/intfloat/multilingual-e5-large (files: config.json, model.safetensors, tokenizer.json, tokenizer_config.json, sentencepiece.bpe.model, special_tokens_map.json) | MIT (model card `license: mit`, checked via the Hub API) | ~560M-parameter multilingual encoder (XLM-R large; 559,890,946 params per the Hub safetensors metadata; 559,891,457 with the 1-logit classification head at load), fine-tuned as a pairwise cross-encoder reranker (rrL) on competition TRAIN pairs only (E021 TR S1, disjoint from stage-2 training and validation S1). | No (general-purpose language model; no entity/business lookup tables) | **COMPLIANT** (MIT, <<8B) |
| Competition Data Intake | E000 | Competition Dataset | 2026 Official | Challenge Portal / `student_resource/` | Competition Provided | Official training & test sets (`train_source[1-3].tsv`, `test_source[1-3].tsv`, `train_ground_truth.tsv`) | Yes (Official competition data only) | **COMPLIANT** |

---

## 3. Repository Inspection & Security Audit

An automated and manual inspection of the entire repository history, commit logs, and source tree was conducted:
1. **Network Calls in Source Code:** Checked `src/` and `experiments/` for `requests`, `urllib`, `http`, `socket`, `httpx`, `aiohttp`, `boto3`, `google`, `openai`, `anthropic`.
   - **Finding:** Exactly **0** network calls found. All candidate generation, feature engineering, and model training scripts execute 100% offline.
2. **External Data Scraping:** Checked git log for `wget`, `curl`, `download`, or scraping scripts.
   - **Finding:** No external web scrapers, curl invocations, or data fetching utilities exist in the repository.
3. **Pretrained Model Check:** Checked for downloaded weights (e.g. Hugging Face checkpoints, sentence-transformers, BERT models).
   - **Finding:** None downloaded yet. The current best system (RECON-08, 89.31% Macro F0.5) is built purely on BM25 lexical retrieval and LightGBM tree models trained from scratch on provided training data.
4. **License Compliance:** All currently utilized libraries (`RapidFuzz`, `LightGBM`, `scikit-learn`, `psutil`) are MIT or BSD licensed, strictly complying with the competition's open-source requirements.


## 4. Compute environments

- Kaggle notebook session (2x Tesla T4 15 GB, 4 vCPU, 31 GB RAM), accessed via the Jupyter proxy. Data uploaded to the session: derived text/pair files built only from the official competition TRAIN data (E017 package: 1,594,626 pair ids + 1,255,347 raw name/address records). No external data. The proxy URL (credential) is kept outside the repository.
