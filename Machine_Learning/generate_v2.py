#!/usr/bin/env python3
"""Generate refactored dataset1_v2.ipynb and dataset2_v2.ipynb.

Reads the text exports (dataset1.txt, dataset2.txt) to extract original cell code,
then reorganises cells per REFACTORING_PLAN.md while preserving all logic & params.
"""
import json, re, os

BASE = r"D:\Queenless\Machine_Learning"

# ═══════════════════════════════════════════════════════════════════════════════
# PARSER
# ═══════════════════════════════════════════════════════════════════════════════

def parse_txt(filepath):
    """Parse dataset.txt → dict {cell_num(1-based): source_string}"""
    with open(filepath, encoding='utf-8') as f:
        text = f.read()
    cells = {}
    parts = re.split(r'\n--- Cell (\d+): (\w+) ---\n', '\n' + text)
    for i in range(1, len(parts) - 2, 3):
        num, typ, body = int(parts[i]), parts[i+1], parts[i+2]
        if '\n### CODE:\n' in body:
            src = body.split('\n### CODE:\n', 1)[1]
        elif '\n### MARKDOWN/RAW:\n' in body:
            src = body.split('\n### MARKDOWN/RAW:\n', 1)[1]
        else:
            src = body
        # Remove output section
        for sep in ['\n\n### OUTPUT:', '\n### OUTPUT:']:
            if sep in src:
                src = src.split(sep, 1)[0]
                break
        cells[num] = src.rstrip('\n')
    return cells


# ═══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def md(s):
    return {"cell_type": "markdown", "metadata": {},
            "source": s.splitlines(True)}

def code(s):
    return {"cell_type": "code", "metadata": {},
            "source": s.splitlines(True), "outputs": [], "execution_count": None}

def save(cells, name):
    path = os.path.join(BASE, name)
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "nbformat": 4, "nbformat_minor": 5
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"Created: {path} ({len(cells)} cells)")

def S(cells, n):
    """Get source from parsed cells by cell number."""
    return cells[n]

def M(cells, nums):
    """Merge sources of multiple cells."""
    return '\n\n'.join(cells[n] for n in nums if n in cells)

def train_cell(models, section_name):
    """Generate a training cell from model specs.
    models: list of (display_name, constructor_str, X_train_var, X_test_var, y_train_var, y_test_var)
    """
    lines = [
        f'print("{"=" * 60}")',
        f'print("  {section_name}")',
        f'print("{"=" * 60}")',
    ]
    for name, ctor, xt, xte, yt, yte in models:
        lines.append(f'train_and_evaluate({ctor},')
        lines.append(f'    {xt}, {yt}, {xte}, {yte},')
        lines.append(f'    "{name}", "{section_name}")')
    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED CONTENT
# ═══════════════════════════════════════════════════════════════════════════════

IMPORTS = """\
import os
import time
import numpy as np
import librosa
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')"""

HELPER = """\
def train_and_evaluate(clf, X_train, y_train, X_test, y_test,
                       model_name="Model", feature_name="Feature"):
    \"\"\"Train model, predict, print accuracy & training time.\"\"\"
    start = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"  {model_name:30s} | Accuracy: {acc*100:.2f}% | Time: {train_time:.2f}s")
    return acc, preds"""

DS1_PATHS = """\
# ============ DATASET PATHS ============
DATASET_NAME = "Dataset 1 - 20k Audio"
BASE_PATH = r"D:\\Queenless\\20k_audio_splitted_dataset"
FEATURE_DIR = r"D:\\Queenless\\features"
LABELS = ['Queen', 'NonQueen']

train_path = os.path.join(BASE_PATH, 'train')
val_path   = os.path.join(BASE_PATH, 'val')
test_path  = os.path.join(BASE_PATH, 'test')
output_dir = FEATURE_DIR
os.makedirs(output_dir, exist_ok=True)
print(f"Dataset: {DATASET_NAME}")
print(f"Labels:  {LABELS}")"""

DS2_PATHS = """\
# ============ DATASET PATHS ============
DATASET_NAME = "Dataset 2 - NuHive/Kaggle"
BASE_PATH = r"D:\\Queenless\\archive\\nuhive_processed"
FEATURE_DIR = r"D:\\Queenless\\kaggle_features"
LABELS = ['bee', 'nobee', 'noqueen']

train_path = os.path.join(BASE_PATH, 'train')
val_path   = os.path.join(BASE_PATH, 'val')
test_path  = os.path.join(BASE_PATH, 'test')
output_dir = FEATURE_DIR
os.makedirs(output_dir, exist_ok=True)
print(f"Dataset: {DATASET_NAME}")
print(f"Labels:  {LABELS}")"""


# ═══════════════════════════════════════════════════════════════════════════════
# PARSE ORIGINALS
# ═══════════════════════════════════════════════════════════════════════════════

ds1 = parse_txt(os.path.join(BASE, "dataset1.txt"))
ds2 = parse_txt(os.path.join(BASE, "dataset2.txt"))


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET 1 V2 CELLS
# ═══════════════════════════════════════════════════════════════════════════════

v1 = []

# ---- Phase 1: Setup (cells 1-4) ----
v1.append(md("# Machine Learning Pipeline - Dataset 1 (20k Audio)"))
v1.append(code(IMPORTS))
v1.append(code(DS1_PATHS))
v1.append(code(HELPER))

# ---- Phase 2: FFT Origin (cells 5-8) ----
# Original: cells 3(md), 4(functions+load+merge), 5(scale), 6-12(models)
v1.append(md("# 1. FFT (Origin)"))
v1.append(code(S(ds1, 4)))   # functions + load + merge (72L)
v1.append(code(S(ds1, 5)))   # scale (3L)
v1.append(code(train_cell([
    ("KNN (raw)",     "KNeighborsClassifier(n_neighbors=55)", "train_features", "test_features", "train_labels", "test_labels"),
    ("KNN (scaled)",  "KNeighborsClassifier(n_neighbors=55)", "train_features_fft_scaled", "test_features_fft_scaled", "train_labels", "test_labels"),
    ("SVM (raw)",     "SVC(C=100, kernel='rbf', degree=3, gamma=0.1, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=42)", "train_features", "test_features", "train_labels", "test_labels"),
    ("SVM (scaled)",  "SVC(C=100, kernel='rbf', degree=3, gamma=0.1, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=42)", "train_features_fft_scaled", "test_features_fft_scaled", "train_labels", "test_labels"),
    ("RF (raw)",      "RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=42, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)", "train_features", "test_features", "train_labels", "test_labels"),
    ("ET (raw)",      "ExtraTreesClassifier(n_estimators=200, criterion='gini', max_depth=30, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=False, oob_score=False, n_jobs=None, random_state=42, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)", "train_features", "test_features", "train_labels", "test_labels"),
    ("ET (scaled)",   "ExtraTreesClassifier(n_estimators=200, criterion='gini', max_depth=30, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=False, oob_score=False, n_jobs=None, random_state=42, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)", "train_features_fft_scaled", "test_features_fft_scaled", "train_labels", "test_labels"),
], "FFT Origin")))

# ---- Phase 3: FFTs (cells 9-13) ----
# Original: 13(md), 14(functions 59L), 15(load_dir 44L), 16(load 6L), 17(shape 1L), 18(scale 3L), 19-34(models)
v1.append(md("# 2. FFTs (Pre-emphasis + Framing + Windowing + FFT)"))
v1.append(code(S(ds1, 14)))                     # signal processing functions (59L)
v1.append(code(M(ds1, [15, 16, 17, 18])))       # load_dir func + load data + shape + scale
v1.append(code(train_cell([
    ("KNN (raw)",     "KNeighborsClassifier(n_neighbors=5)", "fft_train_features", "fft_test_features", "fft_train_labels", "fft_test_labels"),
    ("KNN (scaled)",  "KNeighborsClassifier(n_neighbors=5)", "fft_train_features_scaled", "fft_test_features_scaled", "fft_train_labels", "fft_test_labels"),
    ("SVM (raw)",     "SVC(C=100, kernel='rbf', degree=3, gamma=0.1, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=42)", "fft_train_features", "fft_test_features", "fft_train_labels", "fft_test_labels"),
    ("SVM (scaled)",  "SVC(C=100, kernel='rbf', degree=3, gamma=0.1, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=42)", "fft_train_features_scaled", "fft_test_features_scaled", "fft_train_labels", "fft_test_labels"),
    ("RF (raw)",      "RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=42, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)", "fft_train_features", "fft_test_features", "fft_train_labels", "fft_test_labels"),
    ("ET (raw)",      "ExtraTreesClassifier(n_estimators=200, criterion='gini', max_depth=30, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=False, oob_score=False, n_jobs=None, random_state=42, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)", "fft_train_features", "fft_test_features", "fft_train_labels", "fft_test_labels"),
    ("ET (scaled)",   "ExtraTreesClassifier(n_estimators=200, criterion='gini', max_depth=30, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=False, oob_score=False, n_jobs=None, random_state=42, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)", "fft_train_features_scaled", "fft_test_features_scaled", "fft_train_labels", "fft_test_labels"),
], "FFTs")))

# ---- Phase 4: STFT (cells 14-17) ----
# Original: 51(md), 52(functions+load 65L), 53(shape 1L), 54(scale 3L), 55-57(models)
v1.append(md("# 3. STFT"))
v1.append(code(S(ds1, 52)))                     # functions + load (65L)
v1.append(code(M(ds1, [53, 54])))               # shape + scale
v1.append(code(train_cell([
    ("KNN (raw)",     "KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)", "stft_train_features", "stft_test_features", "stft_train_labels", "stft_test_labels"),
    ("KNN (scaled)",  "KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)", "stft_train_features_scaled", "stft_test_features_scaled", "stft_train_labels", "stft_test_labels"),
    ("SVM (raw)",     "SVC(C=10, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=42)", "stft_train_features", "stft_test_features", "stft_train_labels", "stft_test_labels"),
], "STFT")))

# ---- Phase 5: FFTs + STFT (cells 18-20) ----
# Original: 58(md), 59(combine+scale 27L), 60-63(models)
v1.append(md("# 4. FFTs + STFT"))
v1.append(code(S(ds1, 59)))                     # combine + scale (27L) - SENSITIVE: scales separately
v1.append(code(train_cell([
    ("KNN (scaled)",  "KNeighborsClassifier(n_neighbors=3)", "combined_train_features", "combined_test_features", "combined_train_labels", "combined_test_labels"),
    ("SVM (scaled)",  "SVC(C=100, kernel='rbf', degree=3, gamma=0.1, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=42)", "combined_train_features", "combined_test_features", "combined_train_labels", "combined_test_labels"),
    ("RF (scaled)",   "RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=42, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)", "combined_train_features", "combined_test_features", "combined_train_labels", "combined_test_labels"),
    ("ET (scaled)",   "ExtraTreesClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=False, oob_score=False, n_jobs=None, random_state=42, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)", "combined_train_features", "combined_test_features", "combined_train_labels", "combined_test_labels"),
], "FFTs + STFT")))

# ---- Phase 6: STFTs (cells 21-24) ----
# Original: 64(md), 65(compute_stfts 61L), 66(load_dir+load 63L), 67(shape+scale 5L), 68-73(models)
v1.append(md("# 5. STFTs (Pre-emphasis + STFT per frame)"))
v1.append(code(S(ds1, 65)))                     # compute_stfts functions (61L) - modular
v1.append(code(M(ds1, [66, 67])))               # load + shape + scale
v1.append(code(train_cell([
    ("KNN (raw)",     "KNeighborsClassifier(n_neighbors=5)", "stfts_train_features", "stfts_test_features", "stfts_train_labels", "stfts_test_labels"),
    ("KNN (scaled)",  "KNeighborsClassifier(n_neighbors=5)", "stfts_train_features_scaled", "stfts_test_features_scaled", "stfts_train_labels", "stfts_test_labels"),
    ("SVM (raw)",     "SVC(C=100, kernel='rbf', degree=3, gamma=0.1, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=42)", "stfts_train_features", "stfts_test_features", "stfts_train_labels", "stfts_test_labels"),
    ("SVM (scaled)",  "SVC(C=100, kernel='rbf', degree=3, gamma=0.1, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=42)", "stfts_train_features_scaled", "stfts_test_features_scaled", "stfts_train_labels", "stfts_test_labels"),
    ("RF (raw)",      "RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=42, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)", "stfts_train_features", "stfts_test_features", "stfts_train_labels", "stfts_test_labels"),
    ("ET (raw)",      "ExtraTreesClassifier(n_estimators=200, criterion='gini', max_depth=30, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=False, oob_score=False, n_jobs=None, random_state=42, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)", "stfts_train_features", "stfts_test_features", "stfts_train_labels", "stfts_test_labels"),
], "STFTs")))

# ---- Phase 7: FFTs + STFTs (cells 25-27) ----
# Original: 74(md), 75(big cell 100L - combine+models all in one)
v1.append(md("# 6. FFTs + STFTs"))
v1.append(code(S(ds1, 75)))                     # big consolidated cell (100L)

# ---- Phase 8: CQT (cells 28-31) ----
# Original: 36(md), 37(compute_cqt 17L), 38(load_dir+load 55L), 39(shape 1L), 40(scale 3L), 42,44,46,48,50(models)
v1.append(md("# 7. CQT"))
v1.append(code(S(ds1, 37)))                     # compute_cqt_features (17L)
v1.append(code(M(ds1, [38, 39, 40])))           # load + shape + scale
v1.append(code(train_cell([
    ("KNN (scaled)",  "KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)", "train_features_cqt_scaled", "test_features_cqt_scaled", "train_labels_cqt", "test_labels_cqt"),
    ("SVM (scaled)",  "SVC(C=100, kernel='rbf', gamma='auto', random_state=42)", "train_features_cqt_scaled", "test_features_cqt_scaled", "train_labels_cqt", "test_labels_cqt"),
    ("LR (scaled)",   "LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=500, multi_class='deprecated', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)", "train_features_cqt_scaled", "test_features_cqt_scaled", "train_labels_cqt", "test_labels_cqt"),
    ("RF (raw)",      "RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=False, oob_score=False, n_jobs=None, random_state=42, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)", "train_features_cqt", "test_features_cqt", "train_labels_cqt", "test_labels_cqt"),
    ("ET (scaled)",   "ExtraTreesClassifier(n_estimators=200, criterion='gini', max_depth=30, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=False, oob_score=False, n_jobs=None, random_state=42, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)", "train_features_cqt_scaled", "test_features_cqt_scaled", "train_labels_cqt", "test_labels_cqt"),
], "CQT")))

# ---- Phase 9: CQTs (cells 32-35) ----
# Original: 76(md), 77(compute_cqts 61L), 78(load+merge 64L), 79(shape+scale 5L), 80-85(models)
v1.append(md("# 8. CQTs (Pre-emphasis + CQT)"))
v1.append(code(S(ds1, 77)))                     # compute_cqts_features (61L)
v1.append(code(M(ds1, [78, 79])))               # load + shape + scale
v1.append(code(train_cell([
    ("KNN (raw)",     "KNeighborsClassifier(n_neighbors=5)", "cqts_train_features", "cqts_test_features", "cqts_train_labels", "cqts_test_labels"),
    ("KNN (scaled)",  "KNeighborsClassifier(n_neighbors=5)", "cqts_train_features_scaled", "cqts_test_features_scaled", "cqts_train_labels", "cqts_test_labels"),
    ("SVM (raw)",     "SVC(C=100, kernel='rbf', degree=3, gamma=0.1, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=42)", "cqts_train_features", "cqts_test_features", "cqts_train_labels", "cqts_test_labels"),
    ("SVM (scaled)",  "SVC(C=100, kernel='rbf', degree=3, gamma=0.1, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=42)", "cqts_train_features_scaled", "cqts_test_features_scaled", "cqts_train_labels", "cqts_test_labels"),
    ("RF (raw)",      "RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=42, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)", "cqts_train_features", "cqts_test_features", "cqts_train_labels", "cqts_test_labels"),
    ("ET (raw)",      "ExtraTreesClassifier(n_estimators=200, criterion='gini', max_depth=30, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=False, oob_score=False, n_jobs=None, random_state=42, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)", "cqts_train_features", "cqts_test_features", "cqts_train_labels", "cqts_test_labels"),
], "CQTs")))

# ---- Phase 10: FFTs + CQT (cells 36-40) ----
# Original: 86(md), 87(ALL functions+load+data 230L), 88(shape 1L), 89(scale 3L), 90-98(models)
v1.append(md("# 9. FFTs + CQT"))
v1.append(code(S(ds1, 87)))                     # ALL functions + load + data (230L) - DATASET-SPECIFIC
v1.append(code(M(ds1, [88, 89])))               # shape + scale
v1.append(code(train_cell([
    ("KNN (raw)",     "KNeighborsClassifier(n_neighbors=1)", "train_features", "test_features", "train_labels", "test_labels"),
    ("KNN (scaled)",  "KNeighborsClassifier(n_neighbors=1)", "train_features_scaled", "test_features_scaled", "train_labels", "test_labels"),
    ("SVM (raw)",     "SVC(C=100, kernel='rbf', degree=3, gamma=0.1, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=42)", "train_features", "test_features", "train_labels", "test_labels"),
    ("SVM (scaled)",  "SVC(C=100, kernel='rbf', degree=3, gamma=0.1, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=42)", "train_features_scaled", "test_features_scaled", "train_labels", "test_labels"),
    ("LR (raw)",      "LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=10, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=42, solver='lbfgs', max_iter=1500, multi_class='deprecated', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)", "train_features", "test_features", "train_labels", "test_labels"),
    ("LR (scaled)",   "LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=10, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=42, solver='saga', max_iter=1000, multi_class='deprecated', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)", "train_features_scaled", "test_features_scaled", "train_labels", "test_labels"),
    ("RF (raw)",      "RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=42, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)", "train_features", "test_features", "train_labels", "test_labels"),
    ("ET (raw)",      "ExtraTreesClassifier(n_estimators=200, criterion='gini', max_depth=30, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=False, oob_score=False, n_jobs=None, random_state=42, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)", "train_features", "test_features", "train_labels", "test_labels"),
    ("ET (scaled)",   "ExtraTreesClassifier(n_estimators=200, criterion='gini', max_depth=30, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=False, oob_score=False, n_jobs=None, random_state=42, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)", "train_features_scaled", "test_features_scaled", "train_labels", "test_labels"),
], "FFTs + CQT")))

# ---- Phase 11: PCA (FFTs + CQT) (cells 41-43) ----
# Original: 99(md), 100(separate), 101(pca_fft), 102(pca_cqt), 103(combine), 104(SVM)
v1.append(md("# 10. PCA (FFTs + CQT)"))
v1.append(code(M(ds1, [100, 101, 102, 103])))   # separate + PCA fft + PCA cqt + combine
v1.append(code(train_cell([
    ("SVM (PCA)",     "SVC(C=100, kernel='rbf', degree=3, gamma=0.1, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=500, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=42)", "train_features", "test_features", "train_labels", "test_labels"),
], "PCA (FFTs + CQT)")))

# ---- Phase 11 continued: PCA (FFTs + CQTs + STFT) (cells 44-46) ----
# Original: 106-107(stft redefine+load), 108(shape), 109(pca_stft), 110(combine), 111(shape), 112-114(models), 115(md), 116(big cell)
v1.append(md("# 11. PCA (FFTs + CQTs + STFT)"))
# Use the big consolidated cell from the end which loads all features and trains
v1.append(code(M(ds1, [106, 107])))             # STFT redefine + load
v1.append(code(M(ds1, [108, 109, 110, 111])))   # shape + PCA STFT + combine + shape
v1.append(code(train_cell([
    ("SVM (PCA combined)", "SVC(C=10, kernel='rbf', degree=3, gamma=0.0075, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=500, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=42)", "combined_train_features", "combined_test_features", "train_labels", "test_labels"),
    ("RF (PCA combined)",  "RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=-1, random_state=42, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)", "combined_train_features", "combined_test_features", "train_labels", "test_labels"),
    ("ET (PCA combined)",  "ExtraTreesClassifier(n_estimators=200, criterion='log_loss', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=False, oob_score=False, n_jobs=-1, random_state=42, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)", "combined_train_features", "combined_test_features", "train_labels", "test_labels"),
], "PCA (FFTs + CQTs + STFT)")))

# ---- Final mega PCA cell (optional - the complete reload+PCA+train pipeline) ----
v1.append(md("# 12. PCA Complete Pipeline (FFTs + CQTs + STFT)"))
v1.append(code(S(ds1, 116)))                    # Big consolidated cell (211L)

print(f"Dataset 1: {len(v1)} cells")
save(v1, "dataset1_v2.ipynb")


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET 2 V2 CELLS
# ═══════════════════════════════════════════════════════════════════════════════

v2 = []

# ---- Phase 1: Setup (cells 1-4) ----
v2.append(md("# Machine Learning Pipeline - Dataset 2 (NuHive/Kaggle)"))
v2.append(code(IMPORTS))
v2.append(code(DS2_PATHS))
v2.append(code(HELPER))

# ---- Phase 2: FFT Origin (cells 5-8) ----
# Original: 3(md), 4(functions+load 72L), 5(scale 3L), 6-11(models)
v2.append(md("# 1. FFT (Origin)"))
v2.append(code(S(ds2, 4)))                       # functions + load (72L)
v2.append(code(S(ds2, 5)))                       # scale (3L)
v2.append(code(train_cell([
    ("KNN (raw)",     "KNeighborsClassifier(n_neighbors=19)", "train_features", "test_features", "train_labels", "test_labels"),
    ("KNN (scaled)",  "KNeighborsClassifier(n_neighbors=19)", "train_features_fft_scaled", "test_features_fft_scaled", "train_labels", "test_labels"),
    ("SVM (raw)",     "SVC(C=100, kernel='rbf', degree=3, gamma=0.1, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=42)", "train_features", "test_features", "train_labels", "test_labels"),
    ("SVM (scaled)",  "SVC(C=100, kernel='rbf', degree=3, gamma=0.1, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=42)", "train_features_fft_scaled", "test_features_fft_scaled", "train_labels", "test_labels"),
    ("RF (raw)",      "RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=42, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)", "train_features", "test_features", "train_labels", "test_labels"),
    ("ET (raw)",      "ExtraTreesClassifier(n_estimators=200, criterion='gini', max_depth=30, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=False, oob_score=False, n_jobs=None, random_state=42, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)", "train_features", "test_features", "train_labels", "test_labels"),
], "FFT Origin")))

# ---- Phase 3: FFTs (cells 9-13) ----
# Original: 12(md), 13(functions 56L), 14(load_dir 44L), 15(load 6L), 16(shape 1L), 17(scale 3L), 19-30(models)
v2.append(md("# 2. FFTs (Pre-emphasis + Framing + Windowing + FFT)"))
v2.append(code(S(ds2, 13)))                      # signal processing functions (56L)
v2.append(code(M(ds2, [14, 15, 16, 17])))        # load_dir func + load data + shape + scale
v2.append(code(train_cell([
    ("KNN (raw)",     "KNeighborsClassifier(n_neighbors=5)", "fft_train_features", "fft_test_features", "fft_train_labels", "fft_test_labels"),
    ("KNN (scaled)",  "KNeighborsClassifier(n_neighbors=5)", "fft_train_features_scaled", "fft_test_features_scaled", "fft_train_labels", "fft_test_labels"),
    ("SVM (raw)",     "SVC(C=100, kernel='rbf', degree=3, gamma=0.1, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=42)", "fft_train_features", "fft_test_features", "fft_train_labels", "fft_test_labels"),
    ("SVM (scaled)",  "SVC(C=100, kernel='rbf', degree=3, gamma=0.1, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=42)", "fft_train_features_scaled", "fft_test_features_scaled", "fft_train_labels", "fft_test_labels"),
    ("RF (raw)",      "RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=42, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)", "fft_train_features", "fft_test_features", "fft_train_labels", "fft_test_labels"),
    ("ET (raw)",      "ExtraTreesClassifier(n_estimators=200, criterion='gini', max_depth=30, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=False, oob_score=False, n_jobs=None, random_state=42, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)", "fft_train_features", "fft_test_features", "fft_train_labels", "fft_test_labels"),
], "FFTs")))

# ---- Phase 4: STFT (cells 14-17) ----
# Original: 32(md), 33(stft import+compute 16L), 34(load_dir+load 51L), 35(scale 3L), 36(shape), 37-42(models)
v2.append(md("# 3. STFT"))
v2.append(code(M(ds2, [33, 34])))                # stft functions + load
v2.append(code(M(ds2, [35, 36])))                # scale + shape
v2.append(code(train_cell([
    ("KNN (scaled)",  "KNeighborsClassifier(n_neighbors=25)", "train_features_stft_scaled", "test_features_stft_scaled", "train_labels_stft", "test_labels_stft"),
    ("SVM (raw)",     "SVC(C=10, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=42)", "train_features_stft", "test_features_stft", "train_labels_stft", "test_labels_stft"),
    ("SVM (scaled)",  "SVC(C=10, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=42)", "train_features_stft_scaled", "test_features_stft_scaled", "train_labels_stft", "test_labels_stft"),
    ("RF (raw)",      "RandomForestClassifier(n_estimators=100, random_state=42)", "train_features_stft", "test_features_stft", "train_labels_stft", "test_labels_stft"),
    ("ET (raw)",      "ExtraTreesClassifier(n_estimators=100, random_state=42)", "train_features_stft", "test_features_stft", "train_labels_stft", "test_labels_stft"),
], "STFT")))

# ---- Phase 5: FFTs + STFT (cells 18-20) ----
# Original: 43(md), 44(combine+scale 29L), 45-48(models)
v2.append(md("# 4. FFTs + STFT"))
v2.append(code(S(ds2, 44)))                      # combine + scale - DIFFERENT from ds1! (concat raw → scale)
v2.append(code(train_cell([
    ("KNN (raw)",     "KNeighborsClassifier(n_neighbors=5)", "combined_train_features", "combined_test_features", "combined_train_labels", "combined_test_labels"),
    ("SVM (raw)",     "SVC(C=100, kernel='rbf', degree=3, gamma=0.1, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=42)", "combined_train_features", "combined_test_features", "combined_train_labels", "combined_test_labels"),
    ("RF (raw)",      "RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=42, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)", "combined_train_features", "combined_test_features", "combined_train_labels", "combined_test_labels"),
    ("ET (raw)",      "ExtraTreesClassifier(n_estimators=200, criterion='gini', max_depth=30, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=False, oob_score=False, n_jobs=None, random_state=42, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)", "combined_train_features", "combined_test_features", "combined_train_labels", "combined_test_labels"),
], "FFTs + STFT")))

# ---- Phase 6: STFTs (cells 21-24) ----
# Original: 49(md), 50(compute_stfts 59L), 51(load_dir+load 64L), 52(shape+scale 5L), 53-57(models)
v2.append(md("# 5. STFTs (Pre-emphasis + STFT per frame)"))
v2.append(code(S(ds2, 50)))                      # compute_stfts functions (59L) - INLINE implementation
v2.append(code(M(ds2, [51, 52])))                # load + shape + scale
v2.append(code(train_cell([
    ("KNN (raw)",     "KNeighborsClassifier(n_neighbors=5)", "stfts_train_features", "stfts_test_features", "stfts_train_labels", "stfts_test_labels"),
    ("KNN (scaled)",  "KNeighborsClassifier(n_neighbors=5)", "stfts_train_features_scaled", "stfts_test_features_scaled", "stfts_train_labels", "stfts_test_labels"),
    ("SVM (raw)",     "SVC(C=80, kernel='rbf', degree=3, gamma=0.08, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=42)", "stfts_train_features", "stfts_test_features", "stfts_train_labels", "stfts_test_labels"),
    ("RF (raw)",      "RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=42, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)", "stfts_train_features", "stfts_test_features", "stfts_train_labels", "stfts_test_labels"),
    ("ET (raw)",      "ExtraTreesClassifier(n_estimators=200, criterion='gini', max_depth=30, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=False, oob_score=False, n_jobs=None, random_state=42, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)", "stfts_train_features", "stfts_test_features", "stfts_train_labels", "stfts_test_labels"),
], "STFTs")))

# ---- Phase 7: FFTs + STFTs (cells 25-27) ----
# Original: 58(md), 59(combine 29L), 60-68(models with sub-headers)
v2.append(md("# 6. FFTs + STFTs"))
v2.append(code(S(ds2, 59)))                      # combine (29L)
v2.append(code(train_cell([
    ("KNN (raw)",     "KNeighborsClassifier(n_neighbors=5)", "combined_train_features", "combined_test_features", "combined_train_labels", "combined_test_labels"),
    ("KNN (scaled)",  "KNeighborsClassifier(n_neighbors=5)", "combined_train_features_scaled", "combined_test_features_scaled", "combined_train_labels", "combined_test_labels"),
    ("SVM (raw)",     "SVC(C=100, kernel='rbf', degree=3, gamma=0.1, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=42)", "combined_train_features", "combined_test_features", "combined_train_labels", "combined_test_labels"),
    ("SVM (scaled)",  "SVC(C=100, kernel='rbf', degree=3, gamma=0.1, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=42)", "combined_train_features_scaled", "combined_test_features_scaled", "combined_train_labels", "combined_test_labels"),
    ("RF (raw)",      "RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=42, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)", "combined_train_features", "combined_test_features", "combined_train_labels", "combined_test_labels"),
    ("ET (raw)",      "ExtraTreesClassifier(n_estimators=200, criterion='gini', max_depth=30, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=False, oob_score=False, n_jobs=None, random_state=42, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)", "combined_train_features", "combined_test_features", "combined_train_labels", "combined_test_labels"),
], "FFTs + STFTs")))

# ---- Phase 8: CQT (cells 28-31) ----
# Original: 69(md), 70(compute_cqt 17L), 71(load_dir+load 55L), 72(scale 3L), 74,76,77,79,81,83(models)
v2.append(md("# 7. CQT"))
v2.append(code(S(ds2, 70)))                      # compute_cqt_features (17L)
v2.append(code(M(ds2, [71, 72])))                # load + scale
v2.append(code(train_cell([
    ("KNN (scaled)",  "KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)", "train_features_cqt_scaled", "test_features_cqt_scaled", "train_labels_cqt", "test_labels_cqt"),
    ("SVM (raw)",     "SVC(C=100, kernel='rbf', gamma='auto', random_state=42)", "train_features_cqt", "test_features_cqt", "train_labels_cqt", "test_labels_cqt"),
    ("SVM (scaled)",  "SVC(C=100, kernel='rbf', gamma='auto', random_state=42)", "train_features_cqt_scaled", "test_features_cqt_scaled", "train_labels_cqt", "test_labels_cqt"),
    ("LR (scaled)",   "LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=500, multi_class='deprecated', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)", "train_features_cqt_scaled", "test_features_cqt_scaled", "train_labels_cqt", "test_labels_cqt"),
    ("RF (raw)",      "RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=False, oob_score=False, n_jobs=None, random_state=42, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)", "train_features_cqt", "test_features_cqt", "train_labels_cqt", "test_labels_cqt"),
    ("ET (scaled)",   "ExtraTreesClassifier(n_estimators=200, criterion='gini', max_depth=30, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=False, oob_score=False, n_jobs=None, random_state=42, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)", "train_features_cqt_scaled", "test_features_cqt_scaled", "train_labels_cqt", "test_labels_cqt"),
], "CQT")))

# ---- Phase 9: CQTs (cells 32-35) ----
# Original: 94(md), 95(compute_cqts 61L), 96(load_dir+load 59L), 97(shape+scale 5L), 98-103(models)
v2.append(md("# 8. CQTs (Pre-emphasis + CQT)"))
v2.append(code(S(ds2, 95)))                      # compute_cqts_features (61L)
v2.append(code(M(ds2, [96, 97])))                # load + shape + scale
v2.append(code(train_cell([
    ("KNN (raw)",     "KNeighborsClassifier(n_neighbors=5)", "cqts_train_features", "cqts_test_features", "cqts_train_labels", "cqts_test_labels"),
    ("KNN (scaled)",  "KNeighborsClassifier(n_neighbors=5)", "cqts_train_features_scaled", "cqts_test_features_scaled", "cqts_train_labels", "cqts_test_labels"),
    ("SVM (raw)",     "SVC(C=100, kernel='rbf', degree=3, gamma=0.1, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=42)", "cqts_train_features", "cqts_test_features", "cqts_train_labels", "cqts_test_labels"),
    ("SVM (scaled)",  "SVC(C=100, kernel='rbf', degree=3, gamma=0.1, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=42)", "cqts_train_features_scaled", "cqts_test_features_scaled", "cqts_train_labels", "cqts_test_labels"),
    ("RF (raw)",      "RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=42, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)", "cqts_train_features", "cqts_test_features", "cqts_train_labels", "cqts_test_labels"),
    ("ET (raw)",      "ExtraTreesClassifier(n_estimators=200, criterion='gini', max_depth=30, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=False, oob_score=False, n_jobs=None, random_state=42, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)", "cqts_train_features", "cqts_test_features", "cqts_train_labels", "cqts_test_labels"),
], "CQTs")))

# ---- Phase 10: FFTs + CQT (cells 36-40) ----
# Original: 84(md), 85(ALL functions+load 340L!!!), 86(separate 39L), 87(pca+combine 62L), 88(scale 3L), 89-93(models)
v2.append(md("# 9. FFTs + CQT"))
v2.append(code(S(ds2, 85)))                      # ALL functions + load + data (340L) - DATASET-SPECIFIC!!!
v2.append(code(S(ds2, 86)))                      # separate_fft_cqt_features
v2.append(code(M(ds2, [87, 88])))                # PCA + combine + scale
v2.append(code(train_cell([
    ("KNN (raw)",     "KNeighborsClassifier(n_neighbors=5)", "train_features", "test_features", "train_labels", "test_labels"),
    ("KNN (scaled)",  "KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)", "train_features_scaled", "test_features_scaled", "train_labels", "test_labels"),
    ("SVM (raw)",     "SVC(C=71.19418600172986, kernel='rbf', degree=3, gamma=0.03752055855124281, coef0=0.0, shrinking=True, probability=False, tol=0.0005319450186421158, cache_size=500, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=42)", "train_features", "test_features", "train_labels", "test_labels"),
    ("RF (raw)",      "RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=42, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)", "train_features", "test_features", "train_labels", "test_labels"),
    ("ET (raw)",      "ExtraTreesClassifier(n_estimators=200, criterion='gini', max_depth=30, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=False, oob_score=False, n_jobs=None, random_state=42, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)", "train_features", "test_features", "train_labels", "test_labels"),
], "FFTs + CQT")))

# ---- DS2: FFTs + CQTs combine (extra section in ds2) ----
# Original: 104(md), 105(raw combine+train 82L), 106(PCA combine+train 55L)
v2.append(md("# 10. FFTs + CQTs"))
v2.append(code(S(ds2, 105)))                     # raw combine + train (82L)
v2.append(code(S(ds2, 106)))                     # PCA combine + train (55L)

# ---- Phase 11: PCA (FFTs + CQT + STFT) ----
# Original: 107(md), 108(pca_stft 78L), 109(shapes), 110(combine), 111(scale), 112-116(models)
v2.append(md("# 11. FFTs + CQT + STFT + PCA"))
v2.append(code(M(ds2, [108, 109, 110, 111])))   # PCA STFT + shapes + combine + scale
v2.append(code(train_cell([
    ("KNN (raw)",     "KNeighborsClassifier(n_neighbors=1)", "combined_train_features", "combined_test_features", "train_labels", "test_labels"),
    ("KNN (scaled)",  "KNeighborsClassifier(n_neighbors=3)", "train_features_scaled", "test_features_scaled", "train_labels", "test_labels"),
    ("SVM (raw)",     "SVC(C=71.19418600172986, kernel='rbf', degree=3, gamma=0.03752055855124281, coef0=0.0, shrinking=True, probability=False, tol=0.0005319450186421158, cache_size=500, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=42)", "combined_train_features", "combined_test_features", "train_labels", "test_labels"),
    ("RF (raw)",      "RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=42, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)", "combined_train_features", "combined_test_features", "train_labels", "test_labels"),
    ("ET (raw)",      "ExtraTreesClassifier(n_estimators=200, criterion='gini', max_depth=30, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=False, oob_score=False, n_jobs=None, random_state=42, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)", "combined_train_features", "combined_test_features", "train_labels", "test_labels"),
], "FFTs + CQT + STFT + PCA")))

# ---- Final mega PCA cell ----
v2.append(md("# 12. PCA Complete Pipeline (FFTs + CQTs + STFT)"))
v2.append(code(S(ds2, 118)))                     # Big consolidated cell (198L)

print(f"Dataset 2: {len(v2)} cells")
save(v2, "dataset2_v2.ipynb")

print("\nDone! Check the generated notebooks.")
