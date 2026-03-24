"""
Build dataset2_v2.ipynb from scratch — clean rewrite, NOT copy.
Each cell is thoughtfully rewritten with consistent style.
Dataset 2: NuHive/Kaggle — 3-class (bee, nobee, noqueen).
"""
import json


def md(source: str):
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip().split("\n")}


def code(source: str):
    lines = source.strip().split("\n")
    # Add \n to all lines except the last
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {"cell_type": "code", "metadata": {}, "source": src, "outputs": [], "execution_count": None}


cells = []

# ═══════════════════════════════════════════════════════════
# CELL 1: Title
# ═══════════════════════════════════════════════════════════
cells.append(md("# Machine Learning Pipeline — Dataset 2 (NuHive/Kaggle)"))

# ═══════════════════════════════════════════════════════════
# CELL 2: Imports
# ═══════════════════════════════════════════════════════════
cells.append(code("""\
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
warnings.filterwarnings('ignore')\
"""))

# ═══════════════════════════════════════════════════════════
# CELL 3: Paths & Labels
# ═══════════════════════════════════════════════════════════
cells.append(code("""\
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
print(f"Labels:  {LABELS}")\
"""))

# ═══════════════════════════════════════════════════════════
# CELL 4: Helper
# ═══════════════════════════════════════════════════════════
cells.append(code("""\
def train_and_evaluate(clf, X_train, y_train, X_test, y_test,
                       model_name="Model", feature_name="Feature"):
    \"\"\"Train a classifier, predict on test set, print accuracy & time.\"\"\"
    start = time.time()
    clf.fit(X_train, y_train)
    elapsed = time.time() - start
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"  {model_name:30s} | Acc: {acc*100:.2f}% | Time: {elapsed:.2f}s")
    return acc, preds\
"""))

# ═══════════════════════════════════════════════════════════
# SECTION 1: FFT (Origin) — n_fft=2048 for dataset2
# ═══════════════════════════════════════════════════════════
cells.append(md("# 1. FFT (Origin)"))

cells.append(code("""\
# --- FFT Origin: extract raw FFT magnitude (no pre-emphasis) ---
# Dataset2: n_fft=2048 (vs 512 for dataset1)

def extract_fft_features(file_path, n_fft=2048):
    \"\"\"Load audio and compute raw FFT magnitude (positive freqs only).\"\"\"
    try:
        audio, sr = librosa.load(file_path, sr=None)
        fft = np.fft.fft(audio, n=n_fft)
        fft_magnitude = np.abs(fft[:n_fft // 2])  # positive frequencies
        return fft_magnitude
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def load_dataset_fft(base_path, labels):
    \"\"\"Walk train/val/test splits, extract FFT features, encode labels.\"\"\"
    splits = {}
    for split in ['train', 'val', 'test']:
        features, raw_labels = [], []
        for label in labels:
            folder = os.path.join(base_path, split, label)
            if not os.path.exists(folder):
                continue
            for fname in os.listdir(folder):
                if fname.endswith('.wav'):
                    feat = extract_fft_features(os.path.join(folder, fname), n_fft=2048)
                    if feat is not None:
                        features.append(feat)
                        raw_labels.append(label)
        splits[split] = (np.array(features), raw_labels)

    # Encode labels with a shared LabelEncoder
    le = LabelEncoder()
    all_raw = splits['train'][1] + splits['val'][1] + splits['test'][1]
    le.fit(all_raw)

    result = {}
    for split in ['train', 'val', 'test']:
        feats, raw = splits[split]
        result[split] = (feats, le.transform(raw))
        print(f"  {split}: {feats.shape}")
    return result

print("Loading FFT Origin features...")
fft_data = load_dataset_fft(BASE_PATH, LABELS)\
"""))

cells.append(code("""\
# Merge train + val, scale
fft_train = np.vstack((fft_data['train'][0], fft_data['val'][0]))
fft_train_labels = np.hstack((fft_data['train'][1], fft_data['val'][1]))
fft_test = fft_data['test'][0]
fft_test_labels = fft_data['test'][1]

scaler_fft_origin = StandardScaler()
fft_train_scaled = scaler_fft_origin.fit_transform(fft_train)
fft_test_scaled  = scaler_fft_origin.transform(fft_test)

print(f"Train: {fft_train.shape}, Test: {fft_test.shape}")\
"""))

cells.append(code("""\
print("=== FFT Origin ===")

# KNN raw (k=19) — dataset2 uses k=19 (vs k=55 for dataset1)
train_and_evaluate(KNeighborsClassifier(n_neighbors=19),
                   fft_train, fft_train_labels, fft_test, fft_test_labels,
                   "KNN raw (k=19)", "FFT Origin")

# KNN scaled (k=19)
train_and_evaluate(KNeighborsClassifier(n_neighbors=19),
                   fft_train_scaled, fft_train_labels, fft_test_scaled, fft_test_labels,
                   "KNN scaled (k=19)", "FFT Origin")

# SVM raw (C=100, γ=0.1)
train_and_evaluate(SVC(C=100, kernel='rbf', gamma=0.1, random_state=42),
                   fft_train, fft_train_labels, fft_test, fft_test_labels,
                   "SVM raw (C=100, γ=0.1)", "FFT Origin")

# SVM scaled (C=100, γ=0.1)
train_and_evaluate(SVC(C=100, kernel='rbf', gamma=0.1, random_state=42),
                   fft_train_scaled, fft_train_labels, fft_test_scaled, fft_test_labels,
                   "SVM scaled (C=100, γ=0.1)", "FFT Origin")

# RF raw
train_and_evaluate(RandomForestClassifier(n_estimators=100, random_state=42),
                   fft_train, fft_train_labels, fft_test, fft_test_labels,
                   "RF raw (n=100)", "FFT Origin")

# ET raw (n=200, d=30) — dataset2: NO ET scaled (unlike dataset1)
train_and_evaluate(ExtraTreesClassifier(n_estimators=200, max_depth=30, random_state=42),
                   fft_train, fft_train_labels, fft_test, fft_test_labels,
                   "ET raw (n=200, d=30)", "FFT Origin")\
"""))

# ═══════════════════════════════════════════════════════════
# SECTION 2: FFTs (Pre-emphasis + Framing + Windowing + FFT)
# Dataset2: DIFFERENT framing/fft/log implementations from dataset1
# ═══════════════════════════════════════════════════════════
cells.append(md("# 2. FFTs (Pre-emphasis + Framing + Windowing + FFT)"))

cells.append(code("""\
# --- Signal processing pipeline for FFTs ---
# NOTE: These implementations are DATASET-SPECIFIC.
#   - framing() uses np.zeros + np.concatenate + np.tile, and
#     pad_signal_length = (num_frames - 1) * frame_step + frame_length
#   - fft_frames() uses np.fft.rfft (NOT np.fft.fft + slice)
#   - log uses feature + 1e-8 (NOT np.maximum)

def pre_emphasis(signal_in, pre_emph=0.97):
    \"\"\"Boost high frequencies with first-order high-pass filter.\"\"\"
    return np.append(signal_in[0], signal_in[1:] - pre_emph * signal_in[:-1])

def framing(signal_in, sample_rate, frame_size=0.025, frame_stride=0.01):
    \"\"\"Split signal into overlapping frames using np.zeros/np.concatenate/np.tile.\"\"\"
    frame_length = int(round(frame_size * sample_rate))
    frame_step   = int(round(frame_stride * sample_rate))
    signal_length = len(signal_in)

    num_frames = int(np.ceil(np.abs(signal_length - frame_length) / frame_step)) + 1

    # Dataset2-specific: (num_frames - 1) * frame_step + frame_length
    pad_signal_length = (num_frames - 1) * frame_step + frame_length
    z = np.zeros(pad_signal_length - signal_length)
    pad_signal = np.concatenate([signal_in, z])

    indices = (np.tile(np.arange(0, frame_length), (num_frames, 1))
               + np.tile(np.arange(0, num_frames * frame_step, frame_step),
                         (frame_length, 1)).T)
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    return frames

def windowing(frames):
    \"\"\"Apply Hamming window to each frame.\"\"\"
    frame_length = frames.shape[1]
    return frames * np.hamming(frame_length)

def fft_frames(frames, NFFT=512):
    \"\"\"Compute FFT magnitude per frame — uses np.fft.rfft (not fft+slice).\"\"\"
    return np.absolute(np.fft.rfft(frames, NFFT))\
"""))

cells.append(code("""\
def compute_fft_features(signal_in, sample_rate, frame_size=0.025,
                         frame_stride=0.01, NFFT=512, apply_log=True):
    \"\"\"Full pipeline: pre-emphasis → framing → windowing → FFT → mean.\"\"\"
    emphasized = pre_emphasis(signal_in)
    frames = framing(emphasized, sample_rate, frame_size, frame_stride)
    windowed = windowing(frames)
    mag = fft_frames(windowed, NFFT)
    feature = np.mean(mag, axis=0)          # average across frames
    if apply_log:
        feature = np.log(feature + 1e-8)    # Dataset2: addition (not np.maximum)
    return feature

def load_fft_features_from_directory(directory, sample_rate=22050, NFFT=512,
                                     output_dir=None, dataset_type='train'):
    \"\"\"Extract or load cached FFTs features (pkl prefix: fft3_).\"\"\"
    feat_file  = os.path.join(output_dir, f'fft3_features_{dataset_type}.pkl')
    label_file = os.path.join(output_dir, f'fft3_labels_{dataset_type}.pkl')

    if os.path.exists(feat_file) and os.path.exists(label_file):
        print(f"  Loading {dataset_type} from .pkl ...")
        return joblib.load(feat_file), joblib.load(label_file)

    print(f"  Extracting {dataset_type} ...")
    t0 = time.time()
    features, labels = [], []
    for label in LABELS:
        folder = os.path.join(directory, label)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if not fname.lower().endswith('.wav'):
                continue
            fpath = os.path.join(folder, fname)
            sig, sr = librosa.load(fpath, sr=sample_rate)
            features.append(compute_fft_features(sig, sr, NFFT=NFFT))
            labels.append(label)
    features, labels = np.array(features), np.array(labels)
    if output_dir:
        joblib.dump(features, feat_file)
        joblib.dump(labels, label_file)
    print(f"  Done in {time.time()-t0:.1f}s")
    return features, labels\
"""))

cells.append(code("""\
# Load FFTs features from each split
ffts_train, ffts_train_labels = load_fft_features_from_directory(train_path, NFFT=512, output_dir=output_dir, dataset_type='train')
ffts_val,   ffts_val_labels   = load_fft_features_from_directory(val_path,   NFFT=512, output_dir=output_dir, dataset_type='val')
ffts_test,  ffts_test_labels  = load_fft_features_from_directory(test_path,  NFFT=512, output_dir=output_dir, dataset_type='test')

# Merge train + val
ffts_train = np.vstack((ffts_train, ffts_val))
ffts_train_labels = np.hstack((ffts_train_labels, ffts_val_labels))

# Scale
scaler_ffts = StandardScaler()
ffts_train_scaled = scaler_ffts.fit_transform(ffts_train)
ffts_test_scaled  = scaler_ffts.transform(ffts_test)

print(f"FFTs — Train: {ffts_train.shape}, Test: {ffts_test.shape}")\
"""))

cells.append(code("""\
print("=== FFTs ===")

# KNN raw (k=5) — dataset2 uses k=5
train_and_evaluate(KNeighborsClassifier(n_neighbors=5),
                   ffts_train, ffts_train_labels, ffts_test, ffts_test_labels,
                   "KNN raw (k=5)", "FFTs")

# KNN scaled (k=5)
train_and_evaluate(KNeighborsClassifier(n_neighbors=5),
                   ffts_train_scaled, ffts_train_labels, ffts_test_scaled, ffts_test_labels,
                   "KNN scaled (k=5)", "FFTs")

# SVM raw (C=100, γ=0.1) — dataset2: raw only (no SVM scaled)
train_and_evaluate(SVC(C=100, kernel='rbf', gamma=0.1, random_state=42),
                   ffts_train, ffts_train_labels, ffts_test, ffts_test_labels,
                   "SVM raw (C=100, γ=0.1)", "FFTs")

# RF raw (n=100)
train_and_evaluate(RandomForestClassifier(n_estimators=100, random_state=42),
                   ffts_train, ffts_train_labels, ffts_test, ffts_test_labels,
                   "RF raw (n=100)", "FFTs")

# ET raw (n=200, d=30) — dataset2: NO ET scaled
train_and_evaluate(ExtraTreesClassifier(n_estimators=200, max_depth=30, random_state=42),
                   ffts_train, ffts_train_labels, ffts_test, ffts_test_labels,
                   "ET raw (n=200, d=30)", "FFTs")\
"""))

# ═══════════════════════════════════════════════════════════
# SECTION 3: STFT
# ═══════════════════════════════════════════════════════════
cells.append(md("# 3. STFT"))

cells.append(code("""\
# --- STFT: librosa-based short-time Fourier transform ---

def compute_stft_features(signal_in, sample_rate, n_fft=2048, hop_length=512,
                          apply_log=True):
    \"\"\"Compute STFT magnitude, average over time axis.\"\"\"
    stft_matrix = np.abs(librosa.stft(signal_in, n_fft=n_fft, hop_length=hop_length))
    if apply_log:
        stft_matrix = np.log(stft_matrix + 1e-8)
    return np.mean(stft_matrix, axis=1)   # → shape (n_fft//2+1,)

def load_stft_features_from_directory(directory, sample_rate=22050,
                                      n_fft=2048, hop_length=512,
                                      output_dir=None, dataset_type='train'):
    \"\"\"Extract or load cached STFT features (pkl prefix: stft2_).\"\"\"
    feat_file  = os.path.join(output_dir, f'stft2_features_{dataset_type}.pkl')
    label_file = os.path.join(output_dir, f'stft2_labels_{dataset_type}.pkl')

    if os.path.exists(feat_file) and os.path.exists(label_file):
        print(f"  Loading {dataset_type} from .pkl ...")
        return joblib.load(feat_file), joblib.load(label_file)

    print(f"  Extracting {dataset_type} ...")
    t0 = time.time()
    features, labels = [], []
    for label in LABELS:
        folder = os.path.join(directory, label)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if not fname.lower().endswith('.wav'):
                continue
            fpath = os.path.join(folder, fname)
            sig, sr = librosa.load(fpath, sr=sample_rate)
            features.append(compute_stft_features(sig, sr, n_fft=n_fft,
                                                  hop_length=hop_length))
            labels.append(label)
    features, labels = np.array(features), np.array(labels)
    if output_dir:
        joblib.dump(features, feat_file)
        joblib.dump(labels, label_file)
    print(f"  Done in {time.time()-t0:.1f}s")
    return features, labels\
"""))

cells.append(code("""\
# Load STFT features (n_fft=4096, hop_length=1024)
stft_train, stft_train_labels = load_stft_features_from_directory(train_path, n_fft=4096, hop_length=1024, output_dir=output_dir, dataset_type='train')
stft_val,   stft_val_labels   = load_stft_features_from_directory(val_path,   n_fft=4096, hop_length=1024, output_dir=output_dir, dataset_type='val')
stft_test,  stft_test_labels  = load_stft_features_from_directory(test_path,  n_fft=4096, hop_length=1024, output_dir=output_dir, dataset_type='test')

stft_train = np.vstack((stft_train, stft_val))
stft_train_labels = np.hstack((stft_train_labels, stft_val_labels))

scaler_stft = StandardScaler()
stft_train_scaled = scaler_stft.fit_transform(stft_train)
stft_test_scaled  = scaler_stft.transform(stft_test)

print(f"STFT — Train: {stft_train.shape}, Test: {stft_test.shape}")\
"""))

cells.append(code("""\
print("=== STFT ===")

# KNN raw (k=25) — dataset2 uses k=25 (vs k=3 for dataset1)
train_and_evaluate(KNeighborsClassifier(n_neighbors=25),
                   stft_train, stft_train_labels, stft_test, stft_test_labels,
                   "KNN raw (k=25)", "STFT")

# KNN scaled (k=25)
train_and_evaluate(KNeighborsClassifier(n_neighbors=25),
                   stft_train_scaled, stft_train_labels, stft_test_scaled, stft_test_labels,
                   "KNN scaled (k=25)", "STFT")

# SVM raw (C=10, γ=scale)
train_and_evaluate(SVC(C=10, kernel='rbf', gamma='scale', random_state=42),
                   stft_train, stft_train_labels, stft_test, stft_test_labels,
                   "SVM raw (C=10, γ=scale)", "STFT")

# SVM scaled (C=10, γ=scale)
train_and_evaluate(SVC(C=10, kernel='rbf', gamma='scale', random_state=42),
                   stft_train_scaled, stft_train_labels, stft_test_scaled, stft_test_labels,
                   "SVM scaled (C=10, γ=scale)", "STFT")

# RF raw (n=100) — dataset2 HAS RF on STFT (dataset1 does not)
train_and_evaluate(RandomForestClassifier(n_estimators=100, random_state=42),
                   stft_train, stft_train_labels, stft_test, stft_test_labels,
                   "RF raw (n=100)", "STFT")

# ET scaled (n=100) — dataset2 HAS ET on STFT (dataset1 does not), uses n=100
train_and_evaluate(ExtraTreesClassifier(n_estimators=100, random_state=42),
                   stft_train_scaled, stft_train_labels, stft_test_scaled, stft_test_labels,
                   "ET scaled (n=100)", "STFT")\
"""))

# ═══════════════════════════════════════════════════════════
# SECTION 4: FFTs + STFT (Combine)
# Dataset2: concat raw → scale together (NOT scale separately like dataset1)
# ═══════════════════════════════════════════════════════════
cells.append(md("# 4. FFTs + STFT"))

cells.append(code("""\
# --- FFTs + STFT: concat raw → scale TOGETHER ---
# This is DATASET-SPECIFIC: Dataset2 scales the combined features,
# whereas Dataset1 scales each type separately then concatenates.

# Verify label consistency
if np.array_equal(ffts_train_labels, stft_train_labels) and np.array_equal(ffts_test_labels, stft_test_labels):
    print("✓ Labels are consistent between FFTs and STFT")
    fft_stft_train_labels = ffts_train_labels
    fft_stft_test_labels  = ffts_test_labels
else:
    print("⚠ Warning: Labels mismatch, using FFTs labels")
    fft_stft_train_labels = ffts_train_labels
    fft_stft_test_labels  = ffts_test_labels

# Raw concatenation
fft_stft_train_raw = np.hstack((ffts_train, stft_train))
fft_stft_test_raw  = np.hstack((ffts_test,  stft_test))

# Scale the combined features together
scaler_fft_stft = StandardScaler()
fft_stft_train_scaled = scaler_fft_stft.fit_transform(fft_stft_train_raw)
fft_stft_test_scaled  = scaler_fft_stft.transform(fft_stft_test_raw)

print(f"FFTs+STFT raw    — Train: {fft_stft_train_raw.shape}")
print(f"FFTs+STFT scaled — Train: {fft_stft_train_scaled.shape}")\
"""))

cells.append(code("""\
print("=== FFTs + STFT ===")

# KNN raw (k=5)
train_and_evaluate(KNeighborsClassifier(n_neighbors=5),
                   fft_stft_train_raw, fft_stft_train_labels,
                   fft_stft_test_raw,  fft_stft_test_labels,
                   "KNN raw (k=5)", "FFTs+STFT")

# KNN scaled (k=5)
train_and_evaluate(KNeighborsClassifier(n_neighbors=5),
                   fft_stft_train_scaled, fft_stft_train_labels,
                   fft_stft_test_scaled,  fft_stft_test_labels,
                   "KNN scaled (k=5)", "FFTs+STFT")

# SVM raw (C=100, γ=0.1)
train_and_evaluate(SVC(C=100, kernel='rbf', gamma=0.1, random_state=42),
                   fft_stft_train_raw, fft_stft_train_labels,
                   fft_stft_test_raw,  fft_stft_test_labels,
                   "SVM raw (C=100, γ=0.1)", "FFTs+STFT")

# SVM scaled (C=100, γ=0.1)
train_and_evaluate(SVC(C=100, kernel='rbf', gamma=0.1, random_state=42),
                   fft_stft_train_scaled, fft_stft_train_labels,
                   fft_stft_test_scaled,  fft_stft_test_labels,
                   "SVM scaled (C=100, γ=0.1)", "FFTs+STFT")

# RF raw (n=100)
train_and_evaluate(RandomForestClassifier(n_estimators=100, random_state=42),
                   fft_stft_train_raw, fft_stft_train_labels,
                   fft_stft_test_raw,  fft_stft_test_labels,
                   "RF raw (n=100)", "FFTs+STFT")

# ET raw (n=200, d=30)
train_and_evaluate(ExtraTreesClassifier(n_estimators=200, max_depth=30, random_state=42),
                   fft_stft_train_raw, fft_stft_train_labels,
                   fft_stft_test_raw,  fft_stft_test_labels,
                   "ET raw (n=200, d=30)", "FFTs+STFT")\
"""))

# ═══════════════════════════════════════════════════════════
# SECTION 5: STFTs (Pre-emphasis + STFT per frame)
# Dataset2: INLINE implementation (all operations in one function)
# ═══════════════════════════════════════════════════════════
cells.append(md("# 5. STFTs (Pre-emphasis + STFT per frame)"))

cells.append(code("""\
# --- STFTs: Pre-emphasis → Framing → Windowing → STFT per frame ---
# Dataset2 uses INLINE implementation: all operations inside compute_stfts_features.
# (pre_emphasis, framing, windowing are redefined but NOT called by compute_stfts_features)

def compute_stfts_features(signal_in, sample_rate, frame_size=0.025,
                           frame_stride=0.01, n_fft=2048, apply_log=True):
    \"\"\"Inline pre-emphasis → framing → windowing → STFT per frame → average.\"\"\"
    # Pre-emphasis (inline)
    emphasized = np.append(signal_in[0], signal_in[1:] - 0.97 * signal_in[:-1])

    # Framing (inline)
    frame_length = int(round(frame_size * sample_rate))
    frame_step   = int(round(frame_stride * sample_rate))
    signal_length = len(emphasized)

    num_frames = int(np.ceil(np.abs(signal_length - frame_length) / frame_step)) + 1
    pad_signal_length = num_frames * frame_step + frame_length
    pad_signal = np.pad(emphasized, (0, pad_signal_length - signal_length), mode='constant')

    indices = np.arange(0, frame_length) + np.arange(0, num_frames * frame_step, frame_step)[:, None]
    frames = pad_signal[indices.astype(np.int32)]

    # Windowing (inline)
    frames *= np.hamming(frames.shape[1])

    # STFT per frame: take first time column of each frame's STFT
    mag_frames = []
    for frame in frames:
        stft = np.abs(librosa.stft(frame, n_fft=n_fft))
        mag_frames.append(stft[:, 0])

    mag_frames = np.array(mag_frames)
    feature = np.mean(mag_frames, axis=0)

    if apply_log:
        feature = np.log(np.maximum(feature, 1e-8))
    return feature

def load_stfts_features_from_directory(directory, sample_rate=22050, n_fft=2048,
                                       frame_size=0.025, frame_stride=0.01,
                                       output_dir=None, dataset_type='train'):
    \"\"\"Extract or load cached STFTs features (pkl prefix: stfts3_).\"\"\"
    feat_file  = os.path.join(output_dir, f'stfts3_features_{dataset_type}.pkl')
    label_file = os.path.join(output_dir, f'stfts3_labels_{dataset_type}.pkl')

    if os.path.exists(feat_file) and os.path.exists(label_file):
        print(f"  Loading {dataset_type} from .pkl ...")
        return joblib.load(feat_file), joblib.load(label_file)

    print(f"  Extracting {dataset_type} ...")
    t0 = time.time()
    features, labels = [], []
    for label in LABELS:
        folder = os.path.join(directory, label)
        if not os.path.isdir(folder):
            print(f"  Warning: missing folder {folder}")
            continue
        for fname in os.listdir(folder):
            if not fname.lower().endswith('.wav'):
                continue
            fpath = os.path.join(folder, fname)
            sig, sr = librosa.load(fpath, sr=sample_rate)
            features.append(compute_stfts_features(sig, sr, frame_size=frame_size,
                                                   frame_stride=frame_stride, n_fft=n_fft))
            labels.append(label)
    features, labels = np.array(features), np.array(labels)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(features, feat_file)
        joblib.dump(labels, label_file)
    print(f"  Done in {time.time()-t0:.1f}s")
    return features, labels\
"""))

cells.append(code("""\
stfts_train, stfts_train_labels = load_stfts_features_from_directory(train_path, n_fft=2048, output_dir=output_dir, dataset_type='train')
stfts_val,   stfts_val_labels   = load_stfts_features_from_directory(val_path,   n_fft=2048, output_dir=output_dir, dataset_type='val')
stfts_test,  stfts_test_labels  = load_stfts_features_from_directory(test_path,  n_fft=2048, output_dir=output_dir, dataset_type='test')

stfts_train = np.vstack((stfts_train, stfts_val))
stfts_train_labels = np.hstack((stfts_train_labels, stfts_val_labels))

scaler_stfts = StandardScaler()
stfts_train_scaled = scaler_stfts.fit_transform(stfts_train)
stfts_test_scaled  = scaler_stfts.transform(stfts_test)

print(f"STFTs — Train: {stfts_train.shape}, Test: {stfts_test.shape}")\
"""))

cells.append(code("""\
print("=== STFTs ===")

# KNN raw (k=5)
train_and_evaluate(KNeighborsClassifier(n_neighbors=5),
                   stfts_train, stfts_train_labels, stfts_test, stfts_test_labels,
                   "KNN raw (k=5)", "STFTs")

# KNN scaled (k=5)
train_and_evaluate(KNeighborsClassifier(n_neighbors=5),
                   stfts_train_scaled, stfts_train_labels, stfts_test_scaled, stfts_test_labels,
                   "KNN scaled (k=5)", "STFTs")

# SVM raw (C=80, γ=0.08) — dataset2-specific SVM params
train_and_evaluate(SVC(C=80, kernel='rbf', gamma=0.08, random_state=42),
                   stfts_train, stfts_train_labels, stfts_test, stfts_test_labels,
                   "SVM raw (C=80, γ=0.08)", "STFTs")

# RF raw (n=100)
train_and_evaluate(RandomForestClassifier(n_estimators=100, random_state=42),
                   stfts_train, stfts_train_labels, stfts_test, stfts_test_labels,
                   "RF raw (n=100)", "STFTs")

# ET raw (n=200, d=30)
train_and_evaluate(ExtraTreesClassifier(n_estimators=200, max_depth=30, random_state=42),
                   stfts_train, stfts_train_labels, stfts_test, stfts_test_labels,
                   "ET raw (n=200, d=30)", "STFTs")\
"""))

# ═══════════════════════════════════════════════════════════
# SECTION 6: FFTs + STFTs (Combine)
# ═══════════════════════════════════════════════════════════
cells.append(md("# 6. FFTs + STFTs"))

cells.append(code("""\
# --- Combine FFTs + STFTs ---
ffts_stfts_train = np.hstack((ffts_train, stfts_train))
ffts_stfts_test  = np.hstack((ffts_test,  stfts_test))

scaler_ffts_stfts = StandardScaler()
ffts_stfts_train_scaled = scaler_ffts_stfts.fit_transform(ffts_stfts_train)
ffts_stfts_test_scaled  = scaler_ffts_stfts.transform(ffts_stfts_test)

# Use ffts labels (both come from same audio files)
ffts_stfts_labels_train = ffts_train_labels
ffts_stfts_labels_test  = ffts_test_labels

print(f"FFTs+STFTs — Train: {ffts_stfts_train.shape}, Test: {ffts_stfts_test.shape}")\
"""))

cells.append(code("""\
print("=== FFTs + STFTs ===")

# KNN raw (k=5)
train_and_evaluate(KNeighborsClassifier(n_neighbors=5),
                   ffts_stfts_train, ffts_stfts_labels_train,
                   ffts_stfts_test,  ffts_stfts_labels_test,
                   "KNN raw (k=5)", "FFTs+STFTs")

# KNN scaled (k=5)
train_and_evaluate(KNeighborsClassifier(n_neighbors=5),
                   ffts_stfts_train_scaled, ffts_stfts_labels_train,
                   ffts_stfts_test_scaled,  ffts_stfts_labels_test,
                   "KNN scaled (k=5)", "FFTs+STFTs")

# SVM raw (C=100, γ=0.1)
train_and_evaluate(SVC(C=100, kernel='rbf', gamma=0.1, random_state=42),
                   ffts_stfts_train, ffts_stfts_labels_train,
                   ffts_stfts_test,  ffts_stfts_labels_test,
                   "SVM raw (C=100, γ=0.1)", "FFTs+STFTs")

# SVM scaled (C=100, γ=0.1)
train_and_evaluate(SVC(C=100, kernel='rbf', gamma=0.1, random_state=42),
                   ffts_stfts_train_scaled, ffts_stfts_labels_train,
                   ffts_stfts_test_scaled,  ffts_stfts_labels_test,
                   "SVM scaled (C=100, γ=0.1)", "FFTs+STFTs")

# RF raw (n=100)
train_and_evaluate(RandomForestClassifier(n_estimators=100, random_state=42),
                   ffts_stfts_train, ffts_stfts_labels_train,
                   ffts_stfts_test,  ffts_stfts_labels_test,
                   "RF raw (n=100)", "FFTs+STFTs")

# ET raw (n=200, d=30)
train_and_evaluate(ExtraTreesClassifier(n_estimators=200, max_depth=30, random_state=42),
                   ffts_stfts_train, ffts_stfts_labels_train,
                   ffts_stfts_test,  ffts_stfts_labels_test,
                   "ET raw (n=200, d=30)", "FFTs+STFTs")\
"""))

# ═══════════════════════════════════════════════════════════
# SECTION 7: CQT
# ═══════════════════════════════════════════════════════════
cells.append(md("# 7. CQT"))

cells.append(code("""\
# --- CQT: Constant-Q Transform (no pre-emphasis) ---

def compute_cqt_features(signal_in, sample_rate, hop_length=512, fmin=None,
                         n_bins=84, bins_per_octave=12, apply_log=True):
    \"\"\"Compute CQT magnitude → log → average over time → shape (n_bins,).\"\"\"
    cqt_matrix = np.abs(librosa.cqt(signal_in, sr=sample_rate, hop_length=hop_length,
                                    fmin=fmin, n_bins=n_bins,
                                    bins_per_octave=bins_per_octave))
    if apply_log:
        cqt_matrix = np.log(cqt_matrix + 1e-8)
    return np.mean(cqt_matrix, axis=1)

def load_cqt_features_from_directory(directory, sample_rate=22050, hop_length=512,
                                     fmin=None, n_bins=84, bins_per_octave=12,
                                     output_dir=None, dataset_type='train'):
    \"\"\"Extract or load cached CQT features (pkl prefix: cqt_).\"\"\"
    feat_file  = os.path.join(output_dir, f'cqt_features_{dataset_type}.pkl')
    label_file = os.path.join(output_dir, f'cqt_labels_{dataset_type}.pkl')

    if os.path.exists(feat_file) and os.path.exists(label_file):
        print(f"  Loading {dataset_type} from .pkl ...")
        return joblib.load(feat_file), joblib.load(label_file)

    print(f"  Extracting {dataset_type} ...")
    t0 = time.time()
    features, labels = [], []
    for label in LABELS:
        folder = os.path.join(directory, label)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            sig, sr = librosa.load(fpath, sr=sample_rate)
            features.append(compute_cqt_features(sig, sr, hop_length=hop_length,
                                                 fmin=fmin, n_bins=n_bins,
                                                 bins_per_octave=bins_per_octave))
            labels.append(label)
    features, labels = np.array(features), np.array(labels)
    if output_dir:
        joblib.dump(features, feat_file)
        joblib.dump(labels, label_file)
    print(f"  Done in {time.time()-t0:.1f}s")
    return features, labels\
"""))

cells.append(code("""\
cqt_train, cqt_train_labels = load_cqt_features_from_directory(train_path, hop_length=512, n_bins=84, output_dir=output_dir, dataset_type='train')
cqt_val,   cqt_val_labels   = load_cqt_features_from_directory(val_path,   hop_length=512, n_bins=84, output_dir=output_dir, dataset_type='val')
cqt_test,  cqt_test_labels  = load_cqt_features_from_directory(test_path,  hop_length=512, n_bins=84, output_dir=output_dir, dataset_type='test')

cqt_train = np.vstack((cqt_train, cqt_val))
cqt_train_labels = np.hstack((cqt_train_labels, cqt_val_labels))

scaler_cqt = StandardScaler()
cqt_train_scaled = scaler_cqt.fit_transform(cqt_train)
cqt_test_scaled  = scaler_cqt.transform(cqt_test)

print(f"CQT — Train: {cqt_train.shape}, Test: {cqt_test.shape}")\
"""))

cells.append(code("""\
print("=== CQT ===")

# KNN scaled (k=1) — dataset2 uses k=1 scaled (same as dataset1)
train_and_evaluate(KNeighborsClassifier(n_neighbors=1),
                   cqt_train_scaled, cqt_train_labels, cqt_test_scaled, cqt_test_labels,
                   "KNN scaled (k=1)", "CQT")

# SVM raw (C=100, γ=auto) — dataset2 trains SVM on raw TOO (dataset1: scaled only)
train_and_evaluate(SVC(C=100, kernel='rbf', gamma='auto', random_state=42),
                   cqt_train, cqt_train_labels, cqt_test, cqt_test_labels,
                   "SVM raw (C=100, γ=auto)", "CQT")

# SVM scaled (C=100, γ=auto)
train_and_evaluate(SVC(C=100, kernel='rbf', gamma='auto', random_state=42),
                   cqt_train_scaled, cqt_train_labels, cqt_test_scaled, cqt_test_labels,
                   "SVM scaled (C=100, γ=auto)", "CQT")

# LR scaled (C=1, iter=500)
train_and_evaluate(LogisticRegression(C=1.0, max_iter=500, solver='lbfgs', random_state=None),
                   cqt_train_scaled, cqt_train_labels, cqt_test_scaled, cqt_test_labels,
                   "LR scaled (C=1, iter=500)", "CQT")

# RF raw (bootstrap=False)
train_and_evaluate(RandomForestClassifier(n_estimators=100, bootstrap=False, random_state=42),
                   cqt_train, cqt_train_labels, cqt_test, cqt_test_labels,
                   "RF raw (n=100, no bootstrap)", "CQT")

# ET scaled (n=200, d=30)
train_and_evaluate(ExtraTreesClassifier(n_estimators=200, max_depth=30, random_state=42),
                   cqt_train_scaled, cqt_train_labels, cqt_test_scaled, cqt_test_labels,
                   "ET scaled (n=200, d=30)", "CQT")\
"""))

# ═══════════════════════════════════════════════════════════
# SECTION 8: CQTs (Pre-emphasis + CQT)
# ═══════════════════════════════════════════════════════════
cells.append(md("# 8. CQTs (Pre-emphasis + CQT)"))

cells.append(code("""\
# --- CQTs: Pre-emphasis + CQT ---
# (pre_emphasis already defined above)

def compute_cqts_features(signal_in, sample_rate, frame_size=0.025,
                          frame_stride=0.01, fmin=None, n_bins=84,
                          bins_per_octave=12, apply_log=True):
    \"\"\"Pre-emphasis → CQT on full signal → average over time.\"\"\"
    emphasized = pre_emphasis(signal_in)
    cqt_matrix = np.abs(librosa.cqt(emphasized, sr=sample_rate,
                                    fmin=fmin, n_bins=n_bins,
                                    bins_per_octave=bins_per_octave,
                                    hop_length=512))
    feature = np.mean(cqt_matrix, axis=1)
    if apply_log:
        feature = np.log(np.maximum(feature, 1e-8))
    return feature

def load_cqts_features_from_directory(directory, sample_rate=22050,
                                      frame_size=0.025, frame_stride=0.01,
                                      fmin=None, n_bins=84, bins_per_octave=12,
                                      output_dir=None, dataset_type='train'):
    \"\"\"Extract or load cached CQTs features (pkl prefix: cqts3_).\"\"\"
    feat_file  = os.path.join(output_dir, f'cqts3_features_{dataset_type}.pkl')
    label_file = os.path.join(output_dir, f'cqts3_labels_{dataset_type}.pkl')

    if os.path.exists(feat_file) and os.path.exists(label_file):
        print(f"  Loading {dataset_type} from .pkl ...")
        return joblib.load(feat_file), joblib.load(label_file)

    print(f"  Extracting {dataset_type} ...")
    t0 = time.time()
    features, labels = [], []
    for label in LABELS:
        folder = os.path.join(directory, label)
        if not os.path.isdir(folder):
            print(f"  Warning: missing folder {folder}")
            continue
        for fname in os.listdir(folder):
            if not fname.lower().endswith('.wav'):
                continue
            fpath = os.path.join(folder, fname)
            sig, sr = librosa.load(fpath, sr=sample_rate)
            features.append(compute_cqts_features(sig, sr, frame_size=frame_size,
                                                  frame_stride=frame_stride, fmin=fmin,
                                                  n_bins=n_bins, bins_per_octave=bins_per_octave))
            labels.append(label)
    features, labels = np.array(features), np.array(labels)
    if output_dir:
        joblib.dump(features, feat_file)
        joblib.dump(labels, label_file)
    print(f"  Done in {time.time()-t0:.1f}s")
    return features, labels\
"""))

cells.append(code("""\
cqts_train, cqts_train_labels = load_cqts_features_from_directory(train_path, n_bins=84, output_dir=output_dir, dataset_type='train')
cqts_val,   cqts_val_labels   = load_cqts_features_from_directory(val_path,   n_bins=84, output_dir=output_dir, dataset_type='val')
cqts_test,  cqts_test_labels  = load_cqts_features_from_directory(test_path,  n_bins=84, output_dir=output_dir, dataset_type='test')

cqts_train = np.vstack((cqts_train, cqts_val))
cqts_train_labels = np.hstack((cqts_train_labels, cqts_val_labels))

scaler_cqts = StandardScaler()
cqts_train_scaled = scaler_cqts.fit_transform(cqts_train)
cqts_test_scaled  = scaler_cqts.transform(cqts_test)

print(f"CQTs — Train: {cqts_train.shape}, Test: {cqts_test.shape}")\
"""))

cells.append(code("""\
print("=== CQTs ===")

# KNN raw (k=5)
train_and_evaluate(KNeighborsClassifier(n_neighbors=5),
                   cqts_train, cqts_train_labels, cqts_test, cqts_test_labels,
                   "KNN raw (k=5)", "CQTs")

# KNN scaled (k=5)
train_and_evaluate(KNeighborsClassifier(n_neighbors=5),
                   cqts_train_scaled, cqts_train_labels, cqts_test_scaled, cqts_test_labels,
                   "KNN scaled (k=5)", "CQTs")

# SVM raw (C=100, γ=0.1)
train_and_evaluate(SVC(C=100, kernel='rbf', gamma=0.1, random_state=42),
                   cqts_train, cqts_train_labels, cqts_test, cqts_test_labels,
                   "SVM raw (C=100, γ=0.1)", "CQTs")

# SVM scaled (C=100, γ=0.1)
train_and_evaluate(SVC(C=100, kernel='rbf', gamma=0.1, random_state=42),
                   cqts_train_scaled, cqts_train_labels, cqts_test_scaled, cqts_test_labels,
                   "SVM scaled (C=100, γ=0.1)", "CQTs")

# RF raw (n=100)
train_and_evaluate(RandomForestClassifier(n_estimators=100, random_state=42),
                   cqts_train, cqts_train_labels, cqts_test, cqts_test_labels,
                   "RF raw (n=100)", "CQTs")

# ET raw (n=200, d=30)
train_and_evaluate(ExtraTreesClassifier(n_estimators=200, max_depth=30, random_state=42),
                   cqts_train, cqts_train_labels, cqts_test, cqts_test_labels,
                   "ET raw (n=200, d=30)", "CQTs")\
"""))

# ═══════════════════════════════════════════════════════════
# SECTION 9: FFTs + CQT (Combined extraction)
# Dataset2: COMPLETELY DIFFERENT implementation from dataset1
#   - Bartlett window (not Hamming)
#   - hop_length/overlap framing params
#   - NFFT=2048 (not 512)
#   - n_bins=96 (same)
#   - Bayesian-optimized SVM params
# ═══════════════════════════════════════════════════════════
cells.append(md("# 9. FFTs + CQT (Combined Feature Extraction)"))

cells.append(code("""\
# --- FFTs+CQT: DIFFERENT signal processing from Section 2 ---
# Dataset2 combined pipeline differences:
#   - framing: configurable hop_length/overlap, different pad formula
#   - windowing: Bartlett window (not Hamming)
#   - fft_frames: np.fft.rfft (same as Section 2 for dataset2)
#   - log: feature + 1e-8
#   - NFFT=2048 → feature dim = 1025 (FFT) + 96 (CQT) = 1121

def pre_emphasis_combined(signal_in, pre_emph=0.97):
    \"\"\"Pre-emphasis filter (same as other sections).\"\"\"
    return np.append(signal_in[0], signal_in[1:] - pre_emph * signal_in[:-1])

def framing_combined(signal_in, sample_rate, frame_size=0.025, hop_length=256, overlap=0.6):
    \"\"\"Framing with configurable hop_length/overlap and Bartlett-compatible padding.\"\"\"
    # Determine frame length in samples
    if frame_size <= 1:
        frame_length = int(round(frame_size * sample_rate))
    else:
        frame_length = int(frame_size)

    # Determine hop (step) in samples
    if hop_length is None:
        frame_step = int(frame_length * (1 - overlap))
    elif hop_length <= 1:
        frame_step = int(round(hop_length * sample_rate))
    else:
        frame_step = int(hop_length)
    frame_step = max(1, frame_step)

    signal_length = len(signal_in)
    num_frames = int(np.ceil((signal_length - frame_length) / frame_step)) + 1

    pad_signal_length = (num_frames - 1) * frame_step + frame_length
    pad_zeros = np.zeros(pad_signal_length - signal_length)
    padded_signal = np.concatenate([signal_in, pad_zeros])

    indices = (np.tile(np.arange(0, frame_length), (num_frames, 1))
               + np.tile(np.arange(0, num_frames * frame_step, frame_step),
                         (frame_length, 1)).T)
    indices = np.minimum(indices, len(padded_signal) - 1)
    frames = padded_signal[indices.astype(np.int32, copy=False)]
    return frames

def windowing_combined(frames, window_type='bartlett'):
    \"\"\"Apply configurable window (default: Bartlett for dataset2).\"\"\"
    frame_length = frames.shape[1]
    if window_type == 'hamming':
        window = np.hamming(frame_length)
    elif window_type == 'hanning':
        window = np.hanning(frame_length)
    elif window_type == 'blackman':
        window = np.blackman(frame_length)
    elif window_type == 'bartlett':
        window = np.bartlett(frame_length)
    else:
        window = np.hamming(frame_length)
    return frames * window

def fft_frames_combined(frames, NFFT=512):
    \"\"\"FFT using np.fft.rfft.\"\"\"
    return np.absolute(np.fft.rfft(frames, NFFT))

def compute_fft_features_combined(signal_in, sample_rate, frame_size=0.025,
                                  hop_length=256, overlap=0.6, NFFT=512,
                                  apply_log=True, window_type='bartlett',
                                  normalize=False):
    \"\"\"Full FFT pipeline for combined extraction.\"\"\"
    emphasized = pre_emphasis_combined(signal_in)
    frames = framing_combined(emphasized, sample_rate, frame_size, hop_length, overlap)
    windowed = windowing_combined(frames, window_type)
    mag = fft_frames_combined(windowed, NFFT)
    feature = np.mean(mag, axis=0)
    if apply_log:
        feature = np.log(feature + 1e-8)
    if normalize:
        feature = (feature - np.mean(feature)) / (np.std(feature) + 1e-8)
    return feature

def compute_cqt_features_combined(signal_in, sample_rate, hop_length=512,
                                  fmin=None, n_bins=84, bins_per_octave=12,
                                  apply_log=True, normalize=False):
    \"\"\"CQT feature extraction (with normalize option).\"\"\"
    cqt_matrix = np.abs(librosa.cqt(signal_in, sr=sample_rate, hop_length=hop_length,
                                    fmin=fmin, n_bins=n_bins,
                                    bins_per_octave=bins_per_octave))
    feature = np.mean(cqt_matrix, axis=1)
    if apply_log:
        feature = np.log(feature + 1e-8)
    if normalize:
        feature = (feature - np.mean(feature)) / (np.std(feature) + 1e-8)
    return feature

def compute_combined_features(signal_in, sample_rate, frame_size=0.025,
                              hop_length=512, overlap=0.6, NFFT=512,
                              window_type='bartlett', normalize=False,
                              fmin=None, n_bins=84, bins_per_octave=12,
                              apply_log=True):
    \"\"\"Compute FFT + CQT and concatenate into single vector.\"\"\"
    fft_feat = compute_fft_features_combined(signal_in, sample_rate,
                                            frame_size=frame_size,
                                            hop_length=hop_length,
                                            overlap=overlap, NFFT=NFFT,
                                            window_type=window_type,
                                            apply_log=apply_log,
                                            normalize=normalize)
    cqt_feat = compute_cqt_features_combined(signal_in, sample_rate,
                                            hop_length=hop_length,
                                            fmin=fmin, n_bins=n_bins,
                                            bins_per_octave=bins_per_octave,
                                            apply_log=apply_log,
                                            normalize=normalize)
    return np.concatenate((fft_feat, cqt_feat))\
"""))

cells.append(code("""\
def load_combined_features_from_directory(directory, sample_rate=22050, NFFT=512,
                                         hop_length=512, frame_size=0.025,
                                         overlap=0.6, window_type='bartlett',
                                         fmin=None, n_bins=84, bins_per_octave=12,
                                         normalize=False,
                                         output_dir=None, dataset_type='train'):
    \"\"\"Extract or load cached FFTs+CQT features.
    PKL naming: combined_features_{split}_nfft{NFFT}_hop{hop}_bins{n_bins}.pkl\"\"\"
    hop_str = f"{hop_length}" if hop_length else f"overlap{overlap}"
    feat_file  = os.path.join(output_dir, f'combined_features_{dataset_type}_nfft{NFFT}_hop{hop_str}_bins{n_bins}.pkl')
    label_file = os.path.join(output_dir, f'combined_labels_{dataset_type}_nfft{NFFT}_hop{hop_str}_bins{n_bins}.pkl')

    if os.path.exists(feat_file) and os.path.exists(label_file):
        print(f"  Loading {dataset_type} from .pkl ...")
        return joblib.load(feat_file), joblib.load(label_file)

    print(f"  Extracting {dataset_type} ...")
    t0 = time.time()
    features, labels = [], []
    for label in LABELS:
        folder = os.path.join(directory, label)
        if not os.path.exists(folder):
            print(f"  Warning: {folder} not found, skipping")
            continue
        for fname in os.listdir(folder):
            if not fname.endswith(('.wav', '.mp3', '.flac', '.ogg')):
                continue
            fpath = os.path.join(folder, fname)
            try:
                sig, sr = librosa.load(fpath, sr=sample_rate)
                feat = compute_combined_features(sig, sr, frame_size=frame_size,
                                                hop_length=hop_length,
                                                overlap=overlap, NFFT=NFFT,
                                                window_type=window_type,
                                                normalize=normalize,
                                                fmin=fmin, n_bins=n_bins,
                                                bins_per_octave=bins_per_octave)
                features.append(feat)
                labels.append(label)
            except Exception as e:
                print(f"  Error: {fpath}: {e}")
    features, labels = np.array(features), np.array(labels)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(features, feat_file)
        joblib.dump(labels, label_file)
    print(f"  Done in {time.time()-t0:.1f}s — shape {features.shape}")
    return features, labels\
"""))

cells.append(code("""\
# Load combined FFTs+CQT features (NFFT=2048, hop=512, n_bins=96)
combined_train, combined_train_labels = load_combined_features_from_directory(
    train_path, NFFT=2048, hop_length=512, n_bins=96, output_dir=output_dir, dataset_type='train')
combined_val, combined_val_labels = load_combined_features_from_directory(
    val_path, NFFT=2048, hop_length=512, n_bins=96, output_dir=output_dir, dataset_type='val')
combined_test, combined_test_labels = load_combined_features_from_directory(
    test_path, NFFT=2048, hop_length=512, n_bins=96, output_dir=output_dir, dataset_type='test')

combined_train = np.vstack((combined_train, combined_val))
combined_train_labels = np.hstack((combined_train_labels, combined_val_labels))

scaler_combined = StandardScaler()
combined_train_scaled = scaler_combined.fit_transform(combined_train)
combined_test_scaled  = scaler_combined.transform(combined_test)

print(f"FFTs+CQT — Train: {combined_train.shape}, Test: {combined_test.shape}")\
"""))

cells.append(code("""\
print("=== FFTs + CQT ===")

# KNN raw (k=5) — dataset2 uses k=5 (vs k=1 for dataset1)
train_and_evaluate(KNeighborsClassifier(n_neighbors=5),
                   combined_train, combined_train_labels, combined_test, combined_test_labels,
                   "KNN raw (k=5)", "FFTs+CQT")

# KNN scaled (k=1)
train_and_evaluate(KNeighborsClassifier(n_neighbors=1),
                   combined_train_scaled, combined_train_labels, combined_test_scaled, combined_test_labels,
                   "KNN scaled (k=1)", "FFTs+CQT")

# SVM raw — Bayesian-optimized params (dataset2-specific)
train_and_evaluate(SVC(C=71.19418600172986, kernel='rbf', gamma=0.03752055855124281,
                       tol=0.0005319450186421158, cache_size=500, random_state=42),
                   combined_train, combined_train_labels, combined_test, combined_test_labels,
                   "SVM Bayesian (raw)", "FFTs+CQT")

# RF raw (n=100)
train_and_evaluate(RandomForestClassifier(n_estimators=100, random_state=42),
                   combined_train, combined_train_labels, combined_test, combined_test_labels,
                   "RF raw (n=100)", "FFTs+CQT")

# ET raw (n=200, d=30)
train_and_evaluate(ExtraTreesClassifier(n_estimators=200, max_depth=30, random_state=42),
                   combined_train, combined_train_labels, combined_test, combined_test_labels,
                   "ET raw (n=200, d=30)", "FFTs+CQT")\
"""))

# ═══════════════════════════════════════════════════════════
# SECTION 10: PCA (FFTs + CQT)
# Dataset2: PCA FFT 150, NO PCA on CQT (raw 96) → 246 features
# Dataset1: PCA FFT 130, PCA CQT 80 → 210 features
# ═══════════════════════════════════════════════════════════
cells.append(md("# 10. PCA (FFTs + CQT)"))

cells.append(code("""\
# --- Separate FFT and CQT from the combined features, then PCA FFT ---
# Dataset2: PCA on FFT part only (150 components), keep CQT raw (96)
# → Total: 150 + 96 = 246 features

def separate_fft_cqt(combined_features, nfft=2048, n_bins=96):
    \"\"\"Split combined feature vector into FFT and CQT parts.\"\"\"
    fft_dim = nfft // 2 + 1   # 1025 for NFFT=2048
    if combined_features.ndim == 1:
        return combined_features[:fft_dim], combined_features[fft_dim:]
    return combined_features[:, :fft_dim], combined_features[:, fft_dim:]

# Separate into FFT (1025) and CQT (96) parts
pca_train_fft, pca_train_cqt = separate_fft_cqt(combined_train, nfft=2048)
pca_test_fft,  pca_test_cqt  = separate_fft_cqt(combined_test,  nfft=2048)

print(f"FFT part: {pca_train_fft.shape}, CQT part: {pca_train_cqt.shape}")

# PCA on FFT: 150 components (random_state=53940)
pca_fft = PCA(n_components=150, random_state=53940)
pca_train_fft_reduced = pca_fft.fit_transform(pca_train_fft)
pca_test_fft_reduced  = pca_fft.transform(pca_test_fft)

# Dataset2: NO PCA on CQT — keep raw CQT (96 features)
# Recombine: 150 (PCA FFT) + 96 (raw CQT) = 246 features
pca_train_combined = np.hstack((pca_train_fft_reduced, pca_train_cqt))
pca_test_combined  = np.hstack((pca_test_fft_reduced,  pca_test_cqt))

# Scale
scaler_pca = StandardScaler()
pca_train_scaled = scaler_pca.fit_transform(pca_train_combined)
pca_test_scaled  = scaler_pca.transform(pca_test_combined)

print(f"PCA combined: Train {pca_train_combined.shape}, Test {pca_test_combined.shape}")\
"""))

cells.append(code("""\
print("=== PCA (FFTs + CQT) — 246 features ===")

# KNN raw (k=5)
train_and_evaluate(KNeighborsClassifier(n_neighbors=5),
                   pca_train_combined, combined_train_labels,
                   pca_test_combined,  combined_test_labels,
                   "KNN raw (k=5)", "PCA(FFTs+CQT)")

# KNN scaled (k=1)
train_and_evaluate(KNeighborsClassifier(n_neighbors=1),
                   pca_train_scaled, combined_train_labels,
                   pca_test_scaled,  combined_test_labels,
                   "KNN scaled (k=1)", "PCA(FFTs+CQT)")

# SVM Bayesian (raw)
train_and_evaluate(SVC(C=71.19418600172986, kernel='rbf', gamma=0.03752055855124281,
                       tol=0.0005319450186421158, cache_size=500, random_state=42),
                   pca_train_combined, combined_train_labels,
                   pca_test_combined,  combined_test_labels,
                   "SVM Bayesian (raw)", "PCA(FFTs+CQT)")

# RF raw (n=100)
train_and_evaluate(RandomForestClassifier(n_estimators=100, random_state=42),
                   pca_train_combined, combined_train_labels,
                   pca_test_combined,  combined_test_labels,
                   "RF raw (n=100)", "PCA(FFTs+CQT)")

# ET raw (n=200, d=30)
train_and_evaluate(ExtraTreesClassifier(n_estimators=200, max_depth=30, random_state=42),
                   pca_train_combined, combined_train_labels,
                   pca_test_combined,  combined_test_labels,
                   "ET raw (n=200, d=30)", "PCA(FFTs+CQT)")\
"""))

# ═══════════════════════════════════════════════════════════
# SECTION 11: FFTs + CQTs (Standalone combination — DS2 ONLY)
# Combines FFTs (fft3_ pkl, 257 dim) and CQTs (cqts3_ pkl, 84 dim)
# ═══════════════════════════════════════════════════════════
cells.append(md("# 11. FFTs + CQTs (Standalone)"))

cells.append(code("""\
# --- Combine standalone FFTs and CQTs features ---
# Re-load FFTs (fft3_ pkl, 257 features) and CQTs (cqts3_ pkl, 84 features)
# Concat → 341 features

# Re-load FFTs (reload to ensure fresh data from pkl)
ffts_reload_train, ffts_reload_train_labels = load_fft_features_from_directory(
    train_path, NFFT=512, output_dir=output_dir, dataset_type='train')
ffts_reload_val, _ = load_fft_features_from_directory(
    val_path, NFFT=512, output_dir=output_dir, dataset_type='val')
ffts_reload_test, ffts_reload_test_labels = load_fft_features_from_directory(
    test_path, NFFT=512, output_dir=output_dir, dataset_type='test')
ffts_reload_train = np.vstack((ffts_reload_train, ffts_reload_val))
ffts_reload_train_labels = np.hstack((ffts_reload_train_labels, _))

# Re-load CQTs
cqts_reload_train, cqts_reload_train_labels = load_cqts_features_from_directory(
    train_path, n_bins=84, output_dir=output_dir, dataset_type='train')
cqts_reload_val, _ = load_cqts_features_from_directory(
    val_path, n_bins=84, output_dir=output_dir, dataset_type='val')
cqts_reload_test, cqts_reload_test_labels = load_cqts_features_from_directory(
    test_path, n_bins=84, output_dir=output_dir, dataset_type='test')
cqts_reload_train = np.vstack((cqts_reload_train, cqts_reload_val))
cqts_reload_train_labels = np.hstack((cqts_reload_train_labels, _))

# Verify labels
if np.array_equal(ffts_reload_train_labels, cqts_reload_train_labels):
    sa_train_labels = ffts_reload_train_labels
    sa_test_labels  = ffts_reload_test_labels
else:
    print("⚠ Labels mismatch")
    sa_train_labels = ffts_reload_train_labels
    sa_test_labels  = ffts_reload_test_labels

# Concat FFTs+CQTs → 341 features
sa_train = np.hstack((ffts_reload_train, cqts_reload_train))
sa_test  = np.hstack((ffts_reload_test,  cqts_reload_test))

# Scale
scaler_sa = StandardScaler()
sa_train_scaled = scaler_sa.fit_transform(sa_train)
sa_test_scaled  = scaler_sa.transform(sa_test)

print(f"FFTs+CQTs — Train: {sa_train.shape}, Test: {sa_test.shape}")

# --- Train models (raw + scaled) ---
print("\\n=== FFTs + CQTs (Standalone) ===")
models_sa = {
    "SVM_RBF": SVC(C=71.19418600172986, kernel='rbf', gamma=0.03752055855124281,
                   tol=0.0005319450186421158, cache_size=500, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=7, weights='distance', metric='minkowski', p=2),
    "RF": RandomForestClassifier(n_estimators=400, max_depth=None, n_jobs=-1, random_state=42),
    "ET": ExtraTreesClassifier(n_estimators=600, max_depth=None, n_jobs=-1, random_state=42),
}
for name, clf in models_sa.items():
    train_and_evaluate(clf, sa_train, sa_train_labels, sa_test, sa_test_labels,
                       f"{name} (raw)", "FFTs+CQTs")

# Re-create models for scaled (classifiers are stateful)
models_sa_scaled = {
    "SVM_RBF": SVC(C=71.19418600172986, kernel='rbf', gamma=0.03752055855124281,
                   tol=0.0005319450186421158, cache_size=500, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=7, weights='distance', metric='minkowski', p=2),
    "RF": RandomForestClassifier(n_estimators=400, max_depth=None, n_jobs=-1, random_state=42),
    "ET": ExtraTreesClassifier(n_estimators=600, max_depth=None, n_jobs=-1, random_state=42),
}
for name, clf in models_sa_scaled.items():
    train_and_evaluate(clf, sa_train_scaled, sa_train_labels, sa_test_scaled, sa_test_labels,
                       f"{name} (scaled)", "FFTs+CQTs")\
"""))

cells.append(code("""\
# --- PCA version of FFTs+CQTs ---
# PCA FFTs: 130 components (random_state=42)
# PCA CQTs: 80 components (random_state=42)
# Combined: 210 features

pca_ffts_sa = PCA(n_components=130, random_state=42)
sa_train_ffts_pca = pca_ffts_sa.fit_transform(ffts_reload_train)
sa_test_ffts_pca  = pca_ffts_sa.transform(ffts_reload_test)

pca_cqts_sa = PCA(n_components=80, random_state=42)
sa_train_cqts_pca = pca_cqts_sa.fit_transform(cqts_reload_train)
sa_test_cqts_pca  = pca_cqts_sa.transform(cqts_reload_test)

sa_pca_train = np.hstack((sa_train_ffts_pca, sa_train_cqts_pca))
sa_pca_test  = np.hstack((sa_test_ffts_pca,  sa_test_cqts_pca))

scaler_sa_pca = StandardScaler()
sa_pca_train_scaled = scaler_sa_pca.fit_transform(sa_pca_train)
sa_pca_test_scaled  = scaler_sa_pca.transform(sa_pca_test)

print(f"PCA(FFTs+CQTs) — Train: {sa_pca_train.shape}, Test: {sa_pca_test.shape}")

# Train models (raw + scaled)
print("\\n=== PCA(FFTs + CQTs) — 210 features ===")
models_pca_sa = {
    "SVM_RBF": SVC(C=71.19418600172986, kernel='rbf', gamma=0.03752055855124281,
                   tol=0.0005319450186421158, cache_size=500, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=7, weights='distance', metric='minkowski', p=2),
    "RF": RandomForestClassifier(n_estimators=400, max_depth=None, n_jobs=-1, random_state=42),
    "ET": ExtraTreesClassifier(n_estimators=600, max_depth=None, n_jobs=-1, random_state=42),
}
for name, clf in models_pca_sa.items():
    train_and_evaluate(clf, sa_pca_train, sa_train_labels, sa_pca_test, sa_test_labels,
                       f"{name} (PCA raw)", "FFTs+CQTs")

models_pca_sa_scaled = {
    "SVM_RBF": SVC(C=71.19418600172986, kernel='rbf', gamma=0.03752055855124281,
                   tol=0.0005319450186421158, cache_size=500, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=7, weights='distance', metric='minkowski', p=2),
    "RF": RandomForestClassifier(n_estimators=400, max_depth=None, n_jobs=-1, random_state=42),
    "ET": ExtraTreesClassifier(n_estimators=600, max_depth=None, n_jobs=-1, random_state=42),
}
for name, clf in models_pca_sa_scaled.items():
    train_and_evaluate(clf, sa_pca_train_scaled, sa_train_labels, sa_pca_test_scaled, sa_test_labels,
                       f"{name} (PCA scaled)", "FFTs+CQTs")\
"""))

# ═══════════════════════════════════════════════════════════
# SECTION 12: FFTs + CQT + STFT + PCA (DS2 ONLY)
# Uses PCA features from Section 10 (246 dim) + STFT PCA (80) = 326
# ═══════════════════════════════════════════════════════════
cells.append(md("# 12. FFTs + CQT + STFT + PCA"))

cells.append(code("""\
# --- Add STFT PCA to PCA(FFTs+CQT) features ---
# PCA(FFTs+CQT) already computed: pca_train_combined (246), pca_test_combined (246)
# STFT features already loaded: stft_train, stft_test (from Section 3)

# Re-load STFT for PCA (ensure correct data)
stft_for_pca_train, _ = load_stft_features_from_directory(
    train_path, n_fft=4096, hop_length=1024, output_dir=output_dir, dataset_type='train')
stft_for_pca_val, _ = load_stft_features_from_directory(
    val_path, n_fft=4096, hop_length=1024, output_dir=output_dir, dataset_type='val')
stft_for_pca_test, _ = load_stft_features_from_directory(
    test_path, n_fft=4096, hop_length=1024, output_dir=output_dir, dataset_type='test')
stft_for_pca_train = np.vstack((stft_for_pca_train, stft_for_pca_val))

# PCA on STFT: 80 components (random_state=3951)
pca_stft = PCA(n_components=80, random_state=3951)
stft_pca_train = pca_stft.fit_transform(stft_for_pca_train)
stft_pca_test  = pca_stft.transform(stft_for_pca_test)

# Combine: PCA(FFTs+CQT) [246] + PCA(STFT) [80] = 326 features
fcs_train = np.hstack((pca_train_combined, stft_pca_train))
fcs_test  = np.hstack((pca_test_combined,  stft_pca_test))

# Scale
scaler_fcs = StandardScaler()
fcs_train_scaled = scaler_fcs.fit_transform(fcs_train)
fcs_test_scaled  = scaler_fcs.transform(fcs_test)

print(f"FFTs+CQT+STFT — Train: {fcs_train.shape}, Test: {fcs_test.shape}")\
"""))

cells.append(code("""\
print("=== FFTs + CQT + STFT + PCA — 326 features ===")

# KNN raw (k=1)
train_and_evaluate(KNeighborsClassifier(n_neighbors=1),
                   fcs_train, combined_train_labels,
                   fcs_test,  combined_test_labels,
                   "KNN raw (k=1)", "FFTs+CQT+STFT")

# KNN scaled (k=3)
train_and_evaluate(KNeighborsClassifier(n_neighbors=3),
                   fcs_train_scaled, combined_train_labels,
                   fcs_test_scaled,  combined_test_labels,
                   "KNN scaled (k=3)", "FFTs+CQT+STFT")

# SVM Bayesian (raw)
train_and_evaluate(SVC(C=71.19418600172986, kernel='rbf', gamma=0.03752055855124281,
                       tol=0.0005319450186421158, cache_size=500, random_state=42),
                   fcs_train, combined_train_labels,
                   fcs_test,  combined_test_labels,
                   "SVM Bayesian (raw)", "FFTs+CQT+STFT")

# RF raw (n=100)
train_and_evaluate(RandomForestClassifier(n_estimators=100, random_state=42),
                   fcs_train, combined_train_labels,
                   fcs_test,  combined_test_labels,
                   "RF raw (n=100)", "FFTs+CQT+STFT")

# ET raw (n=200, d=30)
train_and_evaluate(ExtraTreesClassifier(n_estimators=200, max_depth=30, random_state=42),
                   fcs_train, combined_train_labels,
                   fcs_test,  combined_test_labels,
                   "ET raw (n=200, d=30)", "FFTs+CQT+STFT")\
"""))

# ═══════════════════════════════════════════════════════════
# SECTION 13: PCA (FFTs + CQTs + STFT) — Final pipeline
# Re-loads all features, applies PCA to each, combines
# ═══════════════════════════════════════════════════════════
cells.append(md("# 13. PCA (FFTs + CQTs + STFT)"))

cells.append(code("""\
# --- Final pipeline: Re-load everything, PCA each, combine ---
# Re-load FFTs (fft3_ pkl, 257), CQTs (cqts3_ pkl, 84), STFT (stft2_ pkl, 2049)

# FFTs
final_ffts_train, final_train_labels = load_fft_features_from_directory(
    train_path, NFFT=512, output_dir=output_dir, dataset_type='train')
final_ffts_val, _ = load_fft_features_from_directory(
    val_path, NFFT=512, output_dir=output_dir, dataset_type='val')
final_ffts_test, final_test_labels = load_fft_features_from_directory(
    test_path, NFFT=512, output_dir=output_dir, dataset_type='test')
final_ffts_train = np.vstack((final_ffts_train, final_ffts_val))
final_train_labels_full = np.hstack((final_train_labels, _))

# CQTs
final_cqts_train, _ = load_cqts_features_from_directory(
    train_path, n_bins=84, output_dir=output_dir, dataset_type='train')
final_cqts_val, _ = load_cqts_features_from_directory(
    val_path, n_bins=84, output_dir=output_dir, dataset_type='val')
final_cqts_test, _ = load_cqts_features_from_directory(
    test_path, n_bins=84, output_dir=output_dir, dataset_type='test')
final_cqts_train = np.vstack((final_cqts_train, final_cqts_val))

# STFT
final_stft_train, _ = load_stft_features_from_directory(
    train_path, n_fft=4096, hop_length=1024, output_dir=output_dir, dataset_type='train')
final_stft_val, _ = load_stft_features_from_directory(
    val_path, n_fft=4096, hop_length=1024, output_dir=output_dir, dataset_type='val')
final_stft_test, _ = load_stft_features_from_directory(
    test_path, n_fft=4096, hop_length=1024, output_dir=output_dir, dataset_type='test')
final_stft_train = np.vstack((final_stft_train, final_stft_val))

print(f"FFTs: {final_ffts_train.shape}, CQTs: {final_cqts_train.shape}, STFT: {final_stft_train.shape}")

# PCA each (NO random_state — matches original cell 118)
pca_final_ffts = PCA(n_components=130)
final_ffts_pca_train = pca_final_ffts.fit_transform(final_ffts_train)
final_ffts_pca_test  = pca_final_ffts.transform(final_ffts_test)

pca_final_cqts = PCA(n_components=80)
final_cqts_pca_train = pca_final_cqts.fit_transform(final_cqts_train)
final_cqts_pca_test  = pca_final_cqts.transform(final_cqts_test)

pca_final_stft = PCA(n_components=80)
final_stft_pca_train = pca_final_stft.fit_transform(final_stft_train)
final_stft_pca_test  = pca_final_stft.transform(final_stft_test)

# Combine: 130 + 80 + 80 = 290
final_train_all = np.hstack((final_ffts_pca_train, final_cqts_pca_train, final_stft_pca_train))
final_test_all  = np.hstack((final_ffts_pca_test,  final_cqts_pca_test,  final_stft_pca_test))

print(f"Final PCA combined: Train {final_train_all.shape}, Test {final_test_all.shape}")\
"""))

cells.append(code("""\
print("=== PCA (FFTs + CQTs + STFT) — 290 features ===")

# Models: SVM Bayesian, KNN k=7 distance, RF n=400, ET n=600
models_final = {
    "SVM_RBF": SVC(C=71.19418600172986, kernel='rbf', gamma=0.03752055855124281,
                   tol=0.0005319450186421158, cache_size=500, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=7, weights='distance', metric='minkowski', p=2),
    "RF": RandomForestClassifier(n_estimators=400, max_depth=None, n_jobs=-1, random_state=42),
    "ET": ExtraTreesClassifier(n_estimators=600, max_depth=None, n_jobs=-1, random_state=42),
}

results = {}
for name, clf in models_final.items():
    acc, _ = train_and_evaluate(clf, final_train_all, final_train_labels_full,
                                final_test_all, final_test_labels,
                                f"{name}", "PCA Final")
    results[name] = acc

print("\\n=== Leaderboard (sorted by accuracy) ===")
for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name}: {acc*100:.2f}%")\
"""))

# ═══════════════════════════════════════════════════════════
# Build the notebook JSON
# ═══════════════════════════════════════════════════════════

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": cells
}

output_path = r"d:\Queenless\Machine_Learning\dataset2_v2.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Created {output_path}")
print(f"Total cells: {len(cells)}")

# Count by type
md_count = sum(1 for c in cells if c['cell_type'] == 'markdown')
code_count = sum(1 for c in cells if c['cell_type'] == 'code')
print(f"  Markdown: {md_count}")
print(f"  Code: {code_count}")
