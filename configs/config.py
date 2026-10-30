"""
Configuration constants for the Network Intrusion Detection pipeline.
"""

# Random seed for reproducibility across all models and CV splits
SEED = 42

# Number of stratified folds for cross-validation
N_FOLDS = 5

# Dataset ID column
ID_COL = "id"

# Ordered class list (must match competition label strings exactly)
CLASSES = ["Availability", "Benign", "Reconnaissance", "Spoofing"]
LABEL_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_LABEL = {i: c for i, c in enumerate(CLASSES)}

# 15-column flow-geometry signature for Stage-2 pure-signature lookup.
# Including fwd_pkts_payload.tot and bwd_header_size_min prevents label
# collisions between Benign (fwd_tot=840) and Spoofing (fwd_tot=674) TLS flows.
STAGE2_SIGNATURE_COLS = [
    "proto",
    "service",
    "id.resp_p",
    "fwd_pkts_tot",
    "bwd_pkts_tot",
    "fwd_data_pkts_tot",
    "bwd_data_pkts_tot",
    "fwd_pkts_payload.min",
    "fwd_pkts_payload.max",
    "fwd_pkts_payload.tot",
    "bwd_pkts_payload.min",
    "bwd_pkts_payload.max",
    "fwd_init_window_size",
    "bwd_init_window_size",
    "fwd_header_size_min",
]

# Verified Spoofing-exclusive (bwd_min - fwd_min) payload deltas
# for single-packet (1,1) UDP DNS-53 flows derived from RFC-1035
# DNS compression pointer analysis.
SPOOF_DNS11_DMIN = [
    25, 36, 37, 41, 42, 48, 50, 51, 52, 56, 59, 64, 79, 81, 82, 83,
    85, 91, 95, 96, 99, 104, 105, 107, 118, 121, 130, 166, 168, 182, 214, 247,
]

# Data search paths (Kaggle notebook and local development)
DATA_SEARCH_ROOTS = ["/kaggle/input", "./data", "."]

# LightGBM hyperparameters
LGBM_PARAMS = {
    "objective": "multiclass",
    "num_class": 4,
    "n_estimators": 1400,
    "learning_rate": 0.035,
    "num_leaves": 63,
    "max_depth": 8,
    "min_child_samples": 20,
    "subsample": 0.85,
    "colsample_bytree": 0.80,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "verbose": -1,
}

# XGBoost hyperparameters
XGB_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 4,
    "n_estimators": 1400,
    "learning_rate": 0.035,
    "max_depth": 7,
    "min_child_weight": 3,
    "subsample": 0.85,
    "colsample_bytree": 0.80,
    "reg_alpha": 0.1,
    "reg_lambda": 1.2,
    "tree_method": "hist",
    "early_stopping_rounds": 80,
    "eval_metric": "mlogloss",
    "n_jobs": -1,
}

# CatBoost hyperparameters
CATBOOST_PARAMS = {
    "loss_function": "MultiClass",
    "classes_count": 4,
    "iterations": 1600,
    "learning_rate": 0.045,
    "depth": 7,
    "l2_leaf_reg": 3.0,
    "thread_count": -1,
    "verbose": False,
}

# Ensemble blend weights (LightGBM, XGBoost, CatBoost)
ENSEMBLE_WEIGHTS = (0.40, 0.35, 0.25)

# Stage-2 calibration blend alphas
ALPHA_HIGH_CONF = 0.92   # prior weight for flows with prior_confidence >= 0.99
ALPHA_LOW_CONF  = 0.65   # prior weight for flows with 0.95 <= prior_confidence < 0.99
PURE_SIG_WEIGHT = 0.85   # pure 15-column signature blend weight

# Early stopping rounds for GBDT models
EARLY_STOPPING_ROUNDS = 80

# Sample weight exponent for inverse-frequency class balancing
# Exponent 0.35 softens the extreme 108:1 class imbalance
SAMPLE_WEIGHT_EXPONENT = 0.35
