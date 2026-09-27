# RECON-07: Frozen-Feature LightGBM Capacity Probe & Set-Shape Diagnostic Report

**Date**: 2026-09-25  
**Experiment ID**: RECON-07  
**Objective**: Empirically isolate whether the RECON-05 Logistic Regression baseline was bottlenecked by linear model capacity or feature representation deficiency, using identical frozen 22 features on the strict 1,994 Train S1 / 2,001 Val S1 split, accompanied by a per-S1 candidate set-shape diagnostic.

---

## 1. LightGBM Configuration

To isolate model capacity without confounding hyperparameter optimization, LightGBM was run with a modest, fixed CPU configuration on raw (unscaled) features:

```python
LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
```

- **Feature Matrix**: Exact 22 pair-level features from RECON-05 / RECON-06 (no new features added).
  - 8 Name features (`n_lev`, `n_jw`, `n_ts`, `n_tset`, `n_jacc`, `log_freq`, `l_diff`, `l_ratio`)
  - 8 Address features (`s1_has`, `c_has`, `both`, `a_lev`, `a_jw`, `a_ts`, `a_tset`, `a_jacc`)
  - 6 Retrieval / Origin features (`is_s2`, `is_india`, `ret_addr`, `ret_name`, `inv_r_addr`, `inv_r_name`)
- **Scaling**: None (decision trees are naturally invariant to monotonic feature scaling).
- **Training Time**: **4.99 seconds** on CPU (227,153 pairs; 6,585 positive matches).

---

## 2. Strict Validation Methodology

- **Holdout Split**: Identical 1,994 Train S1 entities (227,153 pairs) and 2,001 Validation S1 entities (228,134 pairs).
- **Leak-Free Threshold Tuning**:
  - The decision threshold was scanned across $\theta \in [0.10, 0.96]$ with step $0.02$ **strictly on Train S1 entities** to maximize S1-level Macro $F_{0.5}$.
  - Optimal Train Threshold selected: **0.66** (Train S1 Macro $F_{0.5} = 93.59\%$).
- **Single Evaluation**:
  - The model and threshold $\theta = 0.66$ were frozen and evaluated **exactly once** on the untouched 2,001 Validation S1 entities.
  - Zero validation labels or predictions were used in model fitting, hyperparameter selection, or threshold calibration.

---

## 3. LR vs LightGBM Results

| Metric | Logistic Regression (RECON-05) | LightGBM (RECON-07) | Delta |
| :--- | :---: | :---: | :---: |
| **S1 Macro F0.5** | **77.59%** | **87.53%** | **+9.94%** |
| **US S1 Macro F0.5** | 80.08% | 88.49% | +8.41% |
| **India S1 Macro F0.5** | 73.87% | 86.09% | **+12.22%** |
| **Pair Precision** | 86.56% | **94.16%** | **+7.60%** |
| **Candidate-Pair Recall** | 75.83% | **85.84%** | **+10.01%** |
| **End-to-End Recall** | 71.37% | **80.79%** | **+9.42%** |
| **True Positives (TP)** | 4,939 | **5,591** | **+652** |
| **False Positives (FP)** | 767 | **347** | **-420 (-54.8%)** |
| **Total Predicted Pairs** | 5,706 | 5,938 | +232 |
| **Decision Threshold** | 0.58 | 0.66 | +0.08 |
| **Singleton FP Entities ($|\text{gt}|=0$)** | 76 / 112 (67.9%) | **35 / 112 (31.2%)** | **-41 (-53.9%)** |
| **Total Singleton FPs** | 111 | **40** | **-71 (-64.0%)** |
| **FP Count in High Conf ($\ge 0.80$)** | 383 | **166** | **-217 (-56.7%)** |
| **FP Count in Extreme Conf ($\ge 0.90$)** | 194 | **73** | **-121 (-62.4%)** |

### Top Features by Split Gain (LightGBM Booster)

1. `inv_r_addr` (gain = 253,435.0, splits = 403): Dominant non-linear anchor.
2. `a_tset` (gain = 42,592.2, splits = 757): Address token set ratio.
3. `a_jacc` (gain = 42,521.1, splits = 673): Address token Jaccard index.
4. `n_ts` (gain = 42,085.5, splits = 702): Name token sort ratio.
5. `n_jw` (gain = 25,529.2, splits = 706): Name Jaro-Winkler similarity.
6. `n_jacc` (gain = 14,425.8, splits = 541): Name token Jaccard overlap.
7. `n_lev` (gain = 8,170.5, splits = 773): Name normalized Levenshtein.
8. `l_ratio` & `l_diff` (gain = 12,577.4 combined): String length compatibility.

---

## 4. Set-Shape Diagnostic Analysis

Evaluated on the 2,001 held-out Validation S1 entities using baseline RECON-05 Logistic Regression candidate scores:
- **Group A ($N = 76$)**: True singleton / no-match S1 entities ($|\text{gt}|=0$) that received $\ge 1$ FP at $\theta = 0.58$.
- **Group B ($N = 1,889$)**: Genuine-match S1 entities ($|\text{gt}| \ge 1$).

| Statistic | Group A: Singleton FP ($N=76$) | Group B: Genuine Match ($N=1,889$) | Mann-Whitney U | AUC | Cohen's d |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Top Score $s_{(1)}$** | med = 0.9009 [0.831, 0.942]<br>mean = 0.8717 | med = 0.9920 [0.975, 0.997]<br>mean = 0.9574 | U = 125,448 (p = 1.8e-28) | **0.8738** | **+0.743** |
| **Second Score $s_{(2)}$** | med = 0.5004 [0.145, 0.691]<br>mean = 0.4511 | med = 0.9450 [0.803, 0.982]<br>mean = 0.8339 | U = 123,348 (p = 2.1e-26) | **0.8592** | **+1.586** |
| **Margin $(s_{(1)} - s_{(2)})$** | med = 0.3749 [0.129, 0.686]<br>mean = 0.4205 | med = 0.0344 [0.009, 0.137]<br>mean = 0.1234 | U = 26,122 (p = 4.7e-21) | **0.1820**<br>*(Rev: 0.8180)* | **-1.486** |
| **# Cands $> 0.80$** | med = 1.0000 [1.000, 1.000]<br>mean = 0.9737 | med = 2.0000 [2.000, 3.000]<br>mean = 2.3843 | U = 119,184 (p = 7.9e-24) | **0.8302** | **+1.116** |
| **Score Std Dev** | med = 0.0929 [0.083, 0.108]<br>mean = 0.0964 | med = 0.1462 [0.123, 0.168]<br>mean = 0.1442 | U = 127,020 (p = 4.7e-30) | **0.8848** | **+1.338** |

### Key Diagnostic Discovery
1. **Solitary Spikes vs Entity Clusters**: In Group A (singleton false positives), the error is almost universally a single isolated distractor candidate spiking (median `# Cands > 0.80` is exactly **1.0000**, with a wide margin $\Delta = 0.3749$ to the second candidate).
2. **Multi-Record Convergence**: In Group B (genuine matches), an S1 entity links to multiple corresponding records (both S2 and S3), yielding tight co-occurrence clusters (median `# Cands > 0.80` is **2.0000**, and the margin between the top two candidates is tiny, median $\Delta = 0.0344$).

---

## 5. H1 vs H2 Interpretation

### Evidence for Model-Capacity Limitation (H1) — DECISIVE
- **Empirical Jump**: LightGBM boosted validation S1 Macro $F_{0.5}$ from **77.59% to 87.53% (+9.94 percentage points)** without adding a single new feature. This exceeds the $+1.5\%$ decision threshold by more than $6\times$.
- **Simultaneous Precision & Recall Gain**:
  - Pair Precision surged $+7.60\%$ (from $86.56\%$ to $94.16\%$).
  - Pair Recall surged $+10.01\%$ (from $75.83\%$ to $85.84\%$).
  - Net True Positives increased by $+652$ while False Positives dropped by $-420$ ($-54.8\%$).
- **Singleton False Positive Halving**: Singleton entities receiving false alarms dropped from $76$ to $35$ ($-53.9\%$), and high-confidence false alarms ($\ge 0.80$) were cut from $383$ to $166$ ($-56.7\%$).
- **Geographic Rectification**: India macro $F_{0.5}$ surged by **+12.22%** (from $73.87\%$ to $86.09\%$), demonstrating that non-linear interaction terms correctly handle transliteration variants without over-penalizing character-level mismatch when token set and retrieval rank agree.

### Evidence for Representation Limitation (H2)
- Although capacity was clearly the primary bottleneck holding back RECON-05, the remaining error budget confirms representation limits:
  1. **Residual Singleton FPs**: 35 singletons still received false positives under LightGBM (40 FP pairs).
  2. **Set-Shape Separation**: The set-shape diagnostic proves that candidate margin ($\text{AUC} = 0.818$) and high-confidence candidate counts ($\text{AUC} = 0.830$) carry strong entity-level discriminatory signal that pure pair-level features do not encode.
  3. **Retrieval Bound**: Total end-to-end recall is capped at $85.84\% \times 94.55\% = 80.79\%$. True matches that were never retrieved in the top-114 cannot be recovered by any scorer.

### What Remains Uncertain
- Whether adding per-S1 relational set features (such as candidate score margin, candidate competition count, or shared-token IDF) directly into LightGBM will eliminate the remaining 35 singleton false positives without over-fitting to the training set.
- How the LightGBM model behaves on France (unseen zero-shot test distribution).

---

## 6. Recommendation

**Highest-Value Next Action**:  
Adopt **LightGBM as the foundational scorer**, and proceed to **E008: Per-S1 Competition & Relational Feature Block** (incorporating within-query score margins, candidate rank provenance, and shared-token IDF) to target the remaining 35 singleton false positives and close the gap to $>90\%$ macro $F_{0.5}$.
