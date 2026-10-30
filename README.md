# Network Intrusion Detection — ML Challenge

A leakage-free machine learning pipeline for the **Kaggle Network Intrusion Detection ML Challenge** — a 4-class network traffic classification task evaluated on **Macro F1-Score**.

---

## 1. Competition Overview

| Field | Detail |
|---|---|
| **Task** | Supervised 4-class classification of bidirectional network flow records |
| **Evaluation Metric** | Unweighted Macro F1-Score |
| **Submission Format** | `submission.csv` with columns `id, Target` (33,161 rows) |
| **Input Files** | `train.csv` (84,713 rows × 84 cols), `test.csv` (33,161 rows × 83 cols) |

### Target Classes

| Class | Description |
|---|---|
| `Availability` | DDoS / SYN flood / Slowloris socket exhaustion / HTTP flood |
| `Benign` | Normal client/server traffic (MQTT, HTTP, TLS/SSL, DNS, mDNS, NTP, DHCP, SSH) |
| `Reconnaissance` | Nmap TCP/UDP scans (`-sS`, `-sT`, `-sU -sV`), Nikto HTTP probing, DNS service discovery |
| `Spoofing` | ARP poisoning / MITM active relay across TLS/SSL, HTTP, DNS-53, NBNS-137, mDNS-5353, LLMNR-5355, DHCP |

---

## 2. Problem Statement

Three structural properties of this dataset make standard off-the-shelf GBDT training insufficient:

1. **Severe Class Imbalance (108.9:1 ratio):** `Availability` accounts for 80,072/84,713 rows (94.52%). Because Macro F1 weights all four classes equally, a single misclassification in `Reconnaissance` (735 rows) penalizes the score ~109× more than one in `Availability`.

2. **Severe Covariate Shift:** In `train.csv`, raw TCP port 21 (`tcp | -`) represents 94.08% of flows. In `test.csv`, `tcp | -` drops to 46.03% while `udp | dns` increases from 1.35% to 24.61%, `tcp | mqtt` from 1.12% to 9.59%, and `tcp | http` from 0.79% to 7.91%.

3. **Unseen Destination Ports:** `train.csv` contains 308 distinct destination ports; `test.csv` contains 1,733 — 1,501 of which never appear in training (primarily Nmap scan targets and Windows WebRTC ephemeral ports 50001–50003).

---

## 3. Dataset Description

### Feature Groups (82 raw features)

- **Protocol Identifiers:** `proto` (tcp/udp/icmp), `service` (dns/http/mqtt/ssl/ntp/dhcp/irc/radius/-), `id.orig_p`, `id.resp_p`
- **Packet & Byte Volumes:** `flow_duration`, `fwd/bwd_pkts_tot`, `fwd/bwd_data_pkts_tot`, packet/byte rates, `down_up_ratio`
- **TCP Header & Window:** `fwd/bwd_header_size_{tot,min,max}`, `fwd/bwd_init_window_size`, `fwd_last_window_size`
- **TCP Control Flags:** FIN, SYN, RST, PSH, ACK, URG, CWR, ECE counts
- **Payload Size Distributions:** `fwd/bwd_pkts_payload.{min,max,tot,avg,std}`, `flow_pkts_payload.{min,max,tot,avg,std}`
- **Inter-Arrival Times (IAT):** `fwd/bwd/flow_iat.{min,max,tot,avg,std}`
- **Bulk Transfer & Subflow:** `payload_bytes_per_second`, `fwd/bwd_subflow_{pkts,bytes}`, `fwd/bwd_bulk_{bytes,packets,rate}`
- **Active/Idle Windows:** `active.{min,max,tot,avg,std}`, `idle.{min,max,tot,avg,std}`

---

## 4. Technologies Used

| Category | Libraries |
|---|---|
| **Data Processing** | `numpy`, `pandas` |
| **ML Models** | `lightgbm`, `xgboost`, `catboost` |
| **Validation & Metrics** | `scikit-learn` (`StratifiedKFold`, `f1_score`, `classification_report`) |
| **Optimization** | `scipy.optimize.minimize` (Nelder-Mead) |
| **Runtime** | Python 3.10+ |

---

## 5. Solution Architecture

The pipeline executes 6 sequential stages:

```
Stage-1: Protocol-Invariant Probability Priors
  (TCP window sizes, OS port ranges, mDNS twin detection)
        |
Feature Engineering
  (82 raw + ~40 engineered features, 5 prior columns)
        |
5-Fold Stratified CV — GBDT Ensemble
  LightGBM (40%) + XGBoost (35%) + CatBoost (25%)
        |
Stage-2: Pure Signature Calibration
  (15-column exact lookup + confidence-gated prior blend)
        |
Coordinate Ascent Class Multiplier Optimization
  (Maximize OOF Macro F1 + Nelder-Mead polish)
        |
  submission.csv
```

---

## 6. Data Preprocessing

- **Target Mapping:** String labels mapped to integer indices via `LABEL_TO_IDX = {'Availability': 0, 'Benign': 1, 'Reconnaissance': 2, 'Spoofing': 3}`.
- **Joint Categorical Encoding:** `proto` and `service` are encoded by building a unified vocabulary across both train and test sets; unseen values map to `-1`.
- **Numeric Sanitization:** `np.nan_to_num(..., nan=0.0, posinf=0.0, neginf=0.0)` applied when converting feature DataFrames to float arrays.
- **Sample Weights:** Inverse-frequency class weights with exponent 0.35 to soften the 108:1 class imbalance without over-correcting on the minority classes.

---

## 7. Feature Engineering

`engineer_features()` in `src/pipeline.py` constructs domain-specific features:

| Feature Group | Examples |
|---|---|
| Packet & Byte Asymmetry | `tot_pkts`, `fwd_pkt_share`, `bwd_to_fwd_pkt_ratio`, `fwd_byte_share`, `data_pkt_ratio` |
| Flow Shape Flags | `is_unanswered`, `is_single_pkt_pair`, `is_dual_stack_pair` |
| DNS Payload Deltas | `dns_d_min` (bwd−fwd min payload), `dns_d_max`, `dns_d_tot`, `is_dns_plus16_min`, `is_bwd_min_eq_max` |
| Transductive Flags | `is_mdns115_twin`, `in_dns53_spoof_tuples`, `in_dns53_benign_tuples` |
| OS Port Fingerprints | `orig_p_linux_exclusive` (32768–49151), `orig_p_windows_high` (≥61000), `is_nmap_udp_port` |
| Well-Known Port Flags | One-hot for `id.resp_p` ∈ {21, 22, 53, 80, 123, 137, 443, 1883, 5353, 5355} |
| TCP Window & Header | `hdr_diff_min`, `win_ratio`, `is_fwd_hmin_32`, `is_fwd_hmin_40`, `is_ssl_tot_840` |
| TCP Flag Density | `syn_to_pkt_ratio`, `rst_to_pkt_ratio`, `ack_to_pkt_ratio` |
| Log-Compressed Timing | `log1p_{flow_duration, flow_iat.*, fwd_iat.*, bwd_iat.*, payload_bytes_per_second}` |
| Stage-1 Priors | `prior_avail`, `prior_benign`, `prior_recon`, `prior_spoof`, `prior_confidence` |

---

## 8. Stage-1 Protocol-Invariant Priors

`compute_structural_invariant_priors()` assigns a probability vector to each flow using deterministic protocol and OS fingerprint rules — **before any tree training**:

| Protocol/Port | Rule | Label |
|---|---|---|
| `tcp \| mqtt`, `tcp \| irc` | 100% Benign in training data | Benign |
| `tcp \| http`, `bwd_win=65160`, `fwd_max ∈ {168,186,189}` | Hping3/HTTP flood window | Availability |
| `tcp \| http`, `fwd_win=65476`, odd `orig_p` | Nmap HTTP scan ephemeral port | Reconnaissance |
| `tcp \| ssl`, `bwd_win=26847`, header-gated | MITM TLS relay | Spoofing |
| `tcp \| -`, `resp_p=21`, `fwd_win=64` | Hping3 SYN flood | Availability |
| `tcp \| -`, `fwd_win ∈ {512,1024}` | Nmap SYN/Connect scan | Reconnaissance |
| `udp \| -`, `fwd_pkts=1`, `bwd_pkts=0`, not SSDP/WSD | Unanswered UDP probe | Reconnaissance |
| `udp \| dns`, port 5353, `fwd_max=12` | mDNS Nmap probe | Reconnaissance |
| `udp \| dns`, port 5353, `fwd_min=115`, twin flow | MITM-forwarded mDNS | Spoofing |
| `udp \| dns`, port 53, `(2,2)` flows, `fwd_min ∉ {32,36}` | DNS MITM amplification | Spoofing |
| `udp \| dns`, port 53, `(1,1)` flows, `orig_p ∈ [32768,49152)` | Linux ephemeral MITM | Spoofing |
| `icmp`, `fwd_pkts >= 800` | ICMP flood | Reconnaissance |

The prior vector is injected as 5 additional features into the GBDT feature matrix. `prior_confidence` guides Stage-2 calibration blend weights.

---

## 9. Ensemble Models

Three GBDT families are trained across 5 stratified folds:

| Model | Key Hyperparameters | Blend Weight |
|---|---|---|
| `LightGBM` | `n_estimators=1400, lr=0.035, num_leaves=63, max_depth=8, colsample_bytree=0.80, early_stopping=80` | 40% |
| `XGBoost` | `n_estimators=1400, lr=0.035, max_depth=7, colsample_bytree=0.80, tree_method='hist', early_stopping=80` | 35% |
| `CatBoost` | `iterations=1600, lr=0.045, depth=7, l2_leaf_reg=3.0, early_stopping=80` | 25% |

---

## 10. Validation Strategy

**5-Fold Stratified Cross-Validation** (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`) preserving exact class proportions (~16,943 validation rows per fold).

### Zero-Leakage Protocol

| Stage | Leakage Prevention |
|---|---|
| Stage-1 Priors | `build_dns53_11_tuple_sets` called on `tr_idx` rows only |
| GBDT Training | Fit on `tr_idx`; early stopping evaluated on `va_idx` |
| Stage-2 Signatures | `build_pure_signature_lookup` built from `tr_idx` only |
| Class Multipliers | `optimize_class_multipliers` applied out-of-fold to `va_idx` |

---

## 11. Stage-2 Pure Signature Calibration

`apply_stage2_calibration()` blends the GBDT ensemble output with Stage-1 priors using confidence-gated weights:

| Condition | Prior weight (α) |
|---|---|
| `prior_confidence >= 0.99` | 0.92 |
| `0.95 <= prior_confidence < 0.99` | 0.65 |

Rows whose exact 15-column flow-geometry signature is 100% pure in the training split (`count >= 2`) receive a soft signature override: `0.85 * one_hot + 0.15 * ensemble`. Skipped for `(1,1) UDP DNS-53` and `mDNS-5353` flows where OS port-range rules take precedence.

---

## 12. Class Multiplier Optimization

`optimize_class_multipliers()` finds per-class scaling factors to maximize OOF Macro F1:

$$\hat{y}_i = \arg\max_c \left( P_{i,c} \cdot m_c \right)$$

1. Coordinate grid search: 3 passes × 4 classes × 66 values in [0.50, 1.80]
2. Nelder-Mead polish on the best grid solution
3. Normalization: divide by mean to prevent magnitude drift

This shifts decision boundaries to compensate for minority-class underrepresentation without modifying model weights.

---

## 13. Prediction Workflow

1. Build full DNS-53 (1,1) tuple sets and pure signature lookup from all 84,713 training rows.
2. Accumulate fold-averaged test predictions: `test_prob += model.predict_proba(X_test) / 5`.
3. Blend: `raw_test = 0.40·lgb + 0.35·xgb + 0.25·cat`.
4. Apply Stage-2 calibration with full-train signature lookup.
5. Scale by optimized class multipliers.
6. Map argmax indices → label strings via `IDX_TO_LABEL`.
7. Write `submission.csv`.

---

## 14. Project Structure

```
network-intrusion-detection-ml/
│
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
├── .gitignore
├── main.py                    # Entry point
│
├── src/
│   ├── __init__.py
│   └── pipeline.py            # Complete pipeline
│
└── configs/
    └── config.py              # Hyperparameters and constants
```

---

## 15. Installation

```bash
git clone <repository-url>
cd network-intrusion-detection-ml

python -m venv .venv
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

---

## 16. Execution

Place `train.csv` and `test.csv` in `./data/`, the current directory, or `/kaggle/input/`:

```bash
python main.py
```

The script auto-discovers data files, runs all 6 stages, and writes `submission.csv` to the current directory.

---

## 17. Key Technical Decisions

**Protocol-Theoretic Priors + GBDTs:** Pure GBDTs fail on 1,501 unseen destination ports in `test.csv`. Stage-1 priors encode TCP window and OS port-range invariants as features, enabling tree models to generalize to unseen ports.

**Transductive DNS-53 Tuple Extraction:** OS ephemeral port boundaries classify (1,1) DNS flows without labels. The payload tuple set is built transductively across train+test, restricted to `tr_idx` inside the CV loop to prevent leakage.

**mDNS Twin Detection:** MITM-forwarded mDNS flows appear as synchronized pairs with duration deltas ≤ 0.0002s. Detecting this pair structure identifies Spoofing without relying on training labels.

**15-Column Signature Specificity:** Including `fwd_pkts_payload.tot` prevents collisions between Benign (`fwd_tot=840`) and Spoofing (`fwd_tot=674`) TLS/SSL flows sharing identical packet counts.

**Exponent 0.35 Sample Weights:** Full inverse-frequency weighting (exponent 1.0) over-penalizes Availability predictions. Exponent 0.35 up-weights minority classes just enough to improve Macro F1 without degrading the majority-class boundary.

---

## 18. Limitations

- Competition datasets (`train.csv`, `test.csv`) are not included. Download from the Kaggle competition page.
- Stage-1 invariant rules are derived from this dataset's specific protocol and attack-tool fingerprints; they may not transfer directly to other NIDS datasets.
- No GPU configuration is provided; CPU training is estimated at 15–25 minutes for all 5 folds.
