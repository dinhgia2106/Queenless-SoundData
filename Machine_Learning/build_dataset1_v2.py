"""
Build dataset1_v2.ipynb from scratch — clean rewrite, NOT copy.
Each cell is thoughtfully rewritten with consistent style.
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
cells.append(md("# Machine Learning Pipeline — Dataset 1 (20k Audio)"))

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
# SECTION 1: FFT (Origin)
# ═══════════════════════════════════════════════════════════
cells.append(md("# 1. FFT (Origin)"))

# Cell 6: FFT Origin — functions + load
cells.append(code("""\
# --- FFT Origin: extract raw FFT magnitude (no pre-emphasis) ---

def extract_fft_features(file_path, n_fft=512):
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
                    feat = extract_fft_features(os.path.join(folder, fname), n_fft=512)
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

# Cell 7: FFT Origin — merge train+val, scale
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

# Cell 8: FFT Origin — train models
cells.append(code("""\
print("=== FFT Origin ===" )

# KNN raw (k=55)
train_and_evaluate(KNeighborsClassifier(n_neighbors=55),
                   fft_train, fft_train_labels, fft_test, fft_test_labels,
                   "KNN raw (k=55)", "FFT Origin")

# KNN scaled (k=55)
train_and_evaluate(KNeighborsClassifier(n_neighbors=55),
                   fft_train_scaled, fft_train_labels, fft_test_scaled, fft_test_labels,
                   "KNN scaled (k=55)", "FFT Origin")

# SVM raw
train_and_evaluate(SVC(C=100, kernel='rbf', gamma=0.1, random_state=42),
                   fft_train, fft_train_labels, fft_test, fft_test_labels,
                   "SVM raw (C=100, γ=0.1)", "FFT Origin")

# SVM scaled
train_and_evaluate(SVC(C=100, kernel='rbf', gamma=0.1, random_state=42),
                   fft_train_scaled, fft_train_labels, fft_test_scaled, fft_test_labels,
                   "SVM scaled (C=100, γ=0.1)", "FFT Origin")

# RF raw
train_and_evaluate(RandomForestClassifier(n_estimators=100, random_state=42),
                   fft_train, fft_train_labels, fft_test, fft_test_labels,
                   "RF raw (n=100)", "FFT Origin")

# ET raw
train_and_evaluate(ExtraTreesClassifier(n_estimators=200, max_depth=30, random_state=42),
                   fft_train, fft_train_labels, fft_test, fft_test_labels,
                   "ET raw (n=200, d=30)", "FFT Origin")

# ET scaled (dataset1 only)
train_and_evaluate(ExtraTreesClassifier(n_estimators=200, max_depth=30, random_state=42),
                   fft_train_scaled, fft_train_labels, fft_test_scaled, fft_test_labels,
                   "ET scaled (n=200, d=30)", "FFT Origin")\
"""))

# ═══════════════════════════════════════════════════════════
# SECTION 2: FFTs (Pre-emphasis + Framing + Windowing + FFT)
# ═══════════════════════════════════════════════════════════
cells.append(md("# 2. FFTs (Pre-emphasis + Framing + Windowing + FFT)"))

# Cell 10: Signal processing functions
cells.append(code("""\
# --- Signal processing pipeline for FFTs ---
# NOTE: These implementations are DATASET-SPECIFIC.
#   - framing() uses np.pad + float(np.abs(...)) for num_frames
#   - fft_frames() uses np.fft.fft (not rfft) with [:, :NFFT//2+1]
#   - log uses np.maximum (not addition)

def pre_emphasis(signal_in, pre_emph=0.97):
    \"\"\"Boost high frequencies with first-order high-pass filter.\"\"\"
    return np.append(signal_in[0], signal_in[1:] - pre_emph * signal_in[:-1])

def framing(signal_in, sample_rate, frame_size=0.025, frame_stride=0.01):
    \"\"\"Split signal into overlapping frames using np.pad.\"\"\"
    frame_length = int(round(frame_size * sample_rate))
    frame_step   = int(round(frame_stride * sample_rate))
    signal_length = len(signal_in)

    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step)) + 1

    pad_signal_length = num_frames * frame_step + frame_length
    pad_signal = np.pad(signal_in, (0, pad_signal_length - signal_length), mode='constant')

    indices = (np.arange(0, frame_length)
               + np.arange(0, num_frames * frame_step, frame_step)[:, None])
    frames = pad_signal[indices.astype(np.int32)]
    return frames

def windowing(frames):
    \"\"\"Apply Hamming window to each frame.\"\"\"
    frame_length = frames.shape[1]
    return frames * np.hamming(frame_length)

def fft_frames(frames, NFFT=512):
    \"\"\"Compute FFT magnitude per frame (positive freqs only).\"\"\"
    # Dataset1: uses np.fft.fft then slices [:, :NFFT//2+1]
    return np.abs(np.fft.fft(frames, NFFT))[:, :NFFT // 2 + 1]\
"""))

# Cell 11: compute + load
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
        feature = np.log(np.maximum(feature, 1e-8))  # Dataset1: np.maximum
    return feature

def load_fft_features_from_directory(directory, sample_rate=22050, NFFT=512,
                                     output_dir=None, dataset_type='train'):
    \"\"\"Extract or load cached FFTs features (pkl prefix: ffts3_).\"\"\"
    feat_file  = os.path.join(output_dir, f'ffts3_features_{dataset_type}.pkl')
    label_file = os.path.join(output_dir, f'ffts3_labels_{dataset_type}.pkl')

    if os.path.exists(feat_file) and os.path.exists(label_file):
        print(f"  Loading {dataset_type} from .pkl ...")
        return joblib.load(feat_file), joblib.load(label_file)

    print(f"  Extracting {dataset_type} ...")
    t0 = time.time()
    features, labels = [], []
    for label in LABELS:
        folder = os.path.join(directory, label)
        for fname in os.listdir(folder):
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

# Cell 12: FFTs — Load + merge + scale
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

# Cell 13: FFTs — Train models
cells.append(code("""\
print("=== FFTs ===")

# KNN raw (k=5)
train_and_evaluate(KNeighborsClassifier(n_neighbors=5),
                   ffts_train, ffts_train_labels, ffts_test, ffts_test_labels,
                   "KNN raw (k=5)", "FFTs")

# KNN scaled (k=5)
train_and_evaluate(KNeighborsClassifier(n_neighbors=5),
                   ffts_train_scaled, ffts_train_labels, ffts_test_scaled, ffts_test_labels,
                   "KNN scaled (k=5)", "FFTs")

# SVM raw
train_and_evaluate(SVC(C=100, kernel='rbf', gamma=0.1, random_state=42),
                   ffts_train, ffts_train_labels, ffts_test, ffts_test_labels,
                   "SVM raw (C=100, γ=0.1)", "FFTs")

# SVM scaled
train_and_evaluate(SVC(C=100, kernel='rbf', gamma=0.1, random_state=42),
                   ffts_train_scaled, ffts_train_labels, ffts_test_scaled, ffts_test_labels,
                   "SVM scaled (C=100, γ=0.1)", "FFTs")

# RF raw
train_and_evaluate(RandomForestClassifier(n_estimators=100, random_state=42),
                   ffts_train, ffts_train_labels, ffts_test, ffts_test_labels,
                   "RF raw (n=100)", "FFTs")

# ET raw
train_and_evaluate(ExtraTreesClassifier(n_estimators=200, max_depth=30, random_state=42),
                   ffts_train, ffts_train_labels, ffts_test, ffts_test_labels,
                   "ET raw (n=200, d=30)", "FFTs")

# ET scaled (dataset1 only)
train_and_evaluate(ExtraTreesClassifier(n_estimators=200, max_depth=30, random_state=42),
                   ffts_train_scaled, ffts_train_labels, ffts_test_scaled, ffts_test_labels,
                   "ET scaled (n=200, d=30)", "FFTs")\
"""))

# ═══════════════════════════════════════════════════════════
# SECTION 3: STFT
# ═══════════════════════════════════════════════════════════
cells.append(md("# 3. STFT"))

# Cell 15: STFT — functions
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
        for fname in os.listdir(folder):
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

# Cell 16: STFT — Load + merge + scale
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

# Cell 17: STFT — Train models (NO RF/ET for dataset1!)
cells.append(code("""\
print("=== STFT ===")

# KNN raw (k=3) — dataset1 uses k=3
train_and_evaluate(KNeighborsClassifier(n_neighbors=3),
                   stft_train, stft_train_labels, stft_test, stft_test_labels,
                   "KNN raw (k=3)", "STFT")

# KNN scaled (k=3)
train_and_evaluate(KNeighborsClassifier(n_neighbors=3),
                   stft_train_scaled, stft_train_labels, stft_test_scaled, stft_test_labels,
                   "KNN scaled (k=3)", "STFT")

# SVM raw (C=10, γ=scale) — note: γ='scale', NOT 0.1
train_and_evaluate(SVC(C=10, kernel='rbf', gamma='scale', random_state=42),
                   stft_train, stft_train_labels, stft_test, stft_test_labels,
                   "SVM raw (C=10, γ=scale)", "STFT")

# NOTE: Dataset1 does NOT train RF or ET on STFT\
"""))

# ═══════════════════════════════════════════════════════════
# SECTION 4: FFTs + STFT (Combine)
# ═══════════════════════════════════════════════════════════
cells.append(md("# 4. FFTs + STFT"))

# Cell 19: Combine + Scale
# IMPORTANT: Dataset1 scales SEPARATELY then concatenates!
cells.append(code("""\
# --- FFTs + STFT: scale each type SEPARATELY then concatenate ---
# This is DATASET-SPECIFIC: Dataset1 scales individually, Dataset2 scales after concat.

scaler_fft_for_combo = StandardScaler()
fft_train_sc = scaler_fft_for_combo.fit_transform(ffts_train)
fft_test_sc  = scaler_fft_for_combo.transform(ffts_test)

scaler_stft_for_combo = StandardScaler()
stft_train_sc = scaler_stft_for_combo.fit_transform(stft_train)
stft_test_sc  = scaler_stft_for_combo.transform(stft_test)

# Scaled version (scale separately → concat)
fft_stft_train_scaled = np.concatenate([fft_train_sc, stft_train_sc], axis=1)
fft_stft_test_scaled  = np.concatenate([fft_test_sc,  stft_test_sc],  axis=1)

# Raw version (concat without scaling)
fft_stft_train_raw = np.concatenate([ffts_train, stft_train], axis=1)
fft_stft_test_raw  = np.concatenate([ffts_test,  stft_test],  axis=1)

# Use ffts labels (same as stft labels)
fft_stft_train_labels = ffts_train_labels
fft_stft_test_labels  = ffts_test_labels

print(f"FFTs+STFT scaled — Train: {fft_stft_train_scaled.shape}")
print(f"FFTs+STFT raw    — Train: {fft_stft_train_raw.shape}")\
"""))

# Cell 20: FFTs + STFT — Train models
cells.append(code("""\
print("=== FFTs + STFT ===")

# KNN scaled (k=3)
train_and_evaluate(KNeighborsClassifier(n_neighbors=3),
                   fft_stft_train_scaled, fft_stft_train_labels,
                   fft_stft_test_scaled,  fft_stft_test_labels,
                   "KNN scaled (k=3)", "FFTs+STFT")

# KNN raw (k=3)
train_and_evaluate(KNeighborsClassifier(n_neighbors=3),
                   fft_stft_train_raw, fft_stft_train_labels,
                   fft_stft_test_raw,  fft_stft_test_labels,
                   "KNN raw (k=3)", "FFTs+STFT")

# SVM scaled
train_and_evaluate(SVC(C=100, kernel='rbf', gamma=0.1, random_state=42),
                   fft_stft_train_scaled, fft_stft_train_labels,
                   fft_stft_test_scaled,  fft_stft_test_labels,
                   "SVM scaled (C=100, γ=0.1)", "FFTs+STFT")

# SVM raw
train_and_evaluate(SVC(C=100, kernel='rbf', gamma=0.1, random_state=42),
                   fft_stft_train_raw, fft_stft_train_labels,
                   fft_stft_test_raw,  fft_stft_test_labels,
                   "SVM raw (C=100, γ=0.1)", "FFTs+STFT")

# RF raw
train_and_evaluate(RandomForestClassifier(n_estimators=100, random_state=42),
                   fft_stft_train_raw, fft_stft_train_labels,
                   fft_stft_test_raw,  fft_stft_test_labels,
                   "RF raw (n=100)", "FFTs+STFT")

# ET raw (n=100 for dataset1, NOT 200)
train_and_evaluate(ExtraTreesClassifier(n_estimators=100, random_state=42),
                   fft_stft_train_raw, fft_stft_train_labels,
                   fft_stft_test_raw,  fft_stft_test_labels,
                   "ET raw (n=100)", "FFTs+STFT")\
"""))

# ═══════════════════════════════════════════════════════════
# SECTION 5: STFTs (Pre-emphasis + STFT per frame)
# ═══════════════════════════════════════════════════════════
cells.append(md("# 5. STFTs (Pre-emphasis + STFT per frame)"))

# Cell 22: STFTs — functions
# Dataset1 uses MODULAR approach (calls pre_emphasis, framing, windowing)
cells.append(code("""\
# --- STFTs: Pre-emphasis → Framing → Windowing → STFT per frame ---
# Dataset1 uses MODULAR approach: calls separate functions.
# (pre_emphasis, framing, windowing are already defined in FFTs section)

def compute_stfts_features(signal_in, sample_rate, frame_size=0.025,
                           frame_stride=0.01, n_fft=2048, apply_log=True):
    \"\"\"Apply pre-emphasis → framing → windowing → STFT per frame → average.\"\"\"
    emphasized = pre_emphasis(signal_in)
    frames = framing(emphasized, sample_rate, frame_size, frame_stride)
    windowed = windowing(frames)

    # Compute STFT for each frame (take first column = DC to Nyquist)
    mag_frames = []
    for frame in windowed:
        stft = np.abs(librosa.stft(frame, n_fft=n_fft))
        mag_frames.append(stft[:, 0])   # first time column of each frame's STFT

    mag_frames = np.array(mag_frames)
    feature = np.mean(mag_frames, axis=0)   # average across frames

    if apply_log:
        feature = np.log(np.maximum(feature, 1e-8))
    return feature

def load_stfts_features_from_directory(directory, sample_rate=22050, n_fft=2048,
                                       frame_size=0.025, frame_stride=0.01,
                                       output_dir=None, dataset_type='train'):
    \"\"\"Extract or load cached STFTs features (pkl prefix: stfts_).\"\"\"
    feat_file  = os.path.join(output_dir, f'stfts_features_{dataset_type}.pkl')
    label_file = os.path.join(output_dir, f'stfts_labels_{dataset_type}.pkl')

    if os.path.exists(feat_file) and os.path.exists(label_file):
        print(f"  Loading {dataset_type} from .pkl ...")
        return joblib.load(feat_file), joblib.load(label_file)

    print(f"  Extracting {dataset_type} ...")
    t0 = time.time()
    features, labels = [], []
    for label in LABELS:
        folder = os.path.join(directory, label)
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            sig, sr = librosa.load(fpath, sr=sample_rate)
            features.append(compute_stfts_features(sig, sr, frame_size=frame_size,
                                                   frame_stride=frame_stride, n_fft=n_fft))
            labels.append(label)
    features, labels = np.array(features), np.array(labels)
    if output_dir:
        joblib.dump(features, feat_file)
        joblib.dump(labels, label_file)
    print(f"  Done in {time.time()-t0:.1f}s")
    return features, labels\
"""))

# Cell 23: STFTs — Load + merge + scale
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

# Cell 24: STFTs — Train models
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

# SVM raw (C=100, γ=0.1) — dataset1 specific
train_and_evaluate(SVC(C=100, kernel='rbf', gamma=0.1, random_state=42),
                   stfts_train, stfts_train_labels, stfts_test, stfts_test_labels,
                   "SVM raw (C=100, γ=0.1)", "STFTs")

# SVM scaled (dataset1 only — trains SVM on scaled STFTs)
train_and_evaluate(SVC(C=100, kernel='rbf', gamma=0.1, random_state=42),
                   stfts_train_scaled, stfts_train_labels, stfts_test_scaled, stfts_test_labels,
                   "SVM scaled (C=100, γ=0.1)", "STFTs")

# RF raw
train_and_evaluate(RandomForestClassifier(n_estimators=100, random_state=42),
                   stfts_train, stfts_train_labels, stfts_test, stfts_test_labels,
                   "RF raw (n=100)", "STFTs")

# ET raw
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

# RF raw
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

# Cell 30: CQT — Load + merge + scale
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

# Cell 31: CQT — Train models
cells.append(code("""\
print("=== CQT ===")

# KNN scaled (k=1) — dataset1 uses k=1, scaled only
train_and_evaluate(KNeighborsClassifier(n_neighbors=1),
                   cqt_train_scaled, cqt_train_labels, cqt_test_scaled, cqt_test_labels,
                   "KNN scaled (k=1)", "CQT")

# SVM scaled (C=100, γ=auto) — dataset1 trains SVM on scaled only (no raw SVM)
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
    \"\"\"Extract or load cached CQTs features (pkl prefix: cqts_).\"\"\"
    feat_file  = os.path.join(output_dir, f'cqts_features_{dataset_type}.pkl')
    label_file = os.path.join(output_dir, f'cqts_labels_{dataset_type}.pkl')

    if os.path.exists(feat_file) and os.path.exists(label_file):
        print(f"  Loading {dataset_type} from .pkl ...")
        return joblib.load(feat_file), joblib.load(label_file)

    print(f"  Extracting {dataset_type} ...")
    t0 = time.time()
    features, labels = [], []
    for label in LABELS:
        folder = os.path.join(directory, label)
        for fname in os.listdir(folder):
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

# Cell 34: CQTs — Load + merge + scale
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

# Cell 35: CQTs — Train models
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

# RF raw
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
# ═══════════════════════════════════════════════════════════
cells.append(md("# 9. FFTs + CQT (Combined Feature Extraction)"))

# Cell 37: FFTs+CQT core functions
# IMPORTANT: This section has DIFFERENT implementations from the FFTs section!
# Uses np.zeros+np.append+np.tile for framing, np.fft.rfft for FFT, log(+1e-8)
cells.append(code("""\
# --- FFTs+CQT: DIFFERENT signal processing from Section 2 ---
# This section redefines framing/fft to match the original combined pipeline:
#   - framing: np.zeros + np.append + np.tile (NOT np.pad)
#   - fft_frames: np.fft.rfft (NOT np.fft.fft)
#   - log: fft_feature + 1e-8 (NOT np.maximum)
#   - NFFT = 512 → feature dim = 257 (FFT) + 96 (CQT) = 353

def pre_emphasis_combined(signal_in, pre_emph=0.97):
    \"\"\"Pre-emphasis filter (same as other sections).\"\"\"
    return np.append(signal_in[0], signal_in[1:] - pre_emph * signal_in[:-1])

def framing_combined(signal_in, sample_rate, frame_size=0.025, frame_stride=0.01):
    \"\"\"Framing using np.zeros/np.append/np.tile approach.\"\"\"
    frame_length = int(round(frame_size * sample_rate))
    frame_step   = int(round(frame_stride * sample_rate))
    signal_length = len(signal_in)
    num_frames = int(np.ceil(np.abs(signal_length - frame_length) / frame_step)) + 1

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(signal_in, z)

    indices = (np.tile(np.arange(0, frame_length), (num_frames, 1))
               + np.tile(np.arange(0, num_frames * frame_step, frame_step),
                         (frame_length, 1)).T)
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    return frames

def windowing_combined(frames):
    \"\"\"Hamming window (same as other sections).\"\"\"
    frame_length = frames.shape[1]
    return frames * np.hamming(frame_length)

def fft_frames_combined(frames, NFFT=512):
    \"\"\"FFT using np.fft.rfft (NOT fft+slice like Section 2).\"\"\"
    return np.absolute(np.fft.rfft(frames, NFFT))

def compute_fft_features_combined(signal_in, sample_rate, frame_size=0.025,
                                  frame_stride=0.01, NFFT=512, apply_log=True):
    \"\"\"Full FFT pipeline for combined extraction.\"\"\"
    emphasized = pre_emphasis_combined(signal_in)
    frames = framing_combined(emphasized, sample_rate, frame_size, frame_stride)
    windowed = windowing_combined(frames)
    mag = fft_frames_combined(windowed, NFFT)
    feature = np.mean(mag, axis=0)
    if apply_log:
        feature = np.log(feature + 1e-8)   # Dataset1 FFTs+CQT: addition (not np.maximum)
    return feature

def compute_cqt_features_combined(signal_in, sample_rate, fmin=None,
                                  n_bins=84, bins_per_octave=12, apply_log=True):
    \"\"\"CQT feature extraction (NO hop_length param version).\"\"\"
    cqt_matrix = np.abs(librosa.cqt(signal_in, sr=sample_rate,
                                    fmin=fmin, n_bins=n_bins,
                                    bins_per_octave=bins_per_octave))
    feature = np.mean(cqt_matrix, axis=1)
    if apply_log:
        feature = np.log(feature + 1e-8)
    return feature

def compute_combined_features(signal_in, sample_rate, frame_size=0.025,
                              NFFT=512, fmin=None, n_bins=84,
                              bins_per_octave=12, apply_log=True):
    \"\"\"Compute FFT + CQT and concatenate into single vector.\"\"\"
    fft_feat = compute_fft_features_combined(signal_in, sample_rate,
                                            frame_size=frame_size, NFFT=NFFT,
                                            apply_log=apply_log)
    cqt_feat = compute_cqt_features_combined(signal_in, sample_rate,
                                            fmin=fmin, n_bins=n_bins,
                                            bins_per_octave=bins_per_octave,
                                            apply_log=apply_log)
    return np.concatenate((fft_feat, cqt_feat))\
"""))

# Cell 38: FFTs+CQT load function
cells.append(code("""\
def load_combined_features_from_directory(directory, sample_rate=22050, NFFT=512,
                                         frame_size=0.025, fmin=None, n_bins=84,
                                         bins_per_octave=12, output_dir=None,
                                         dataset_type='train'):
    \"\"\"Extract or load cached FFTs+CQT features.
    PKL prefix: combined_features2_{split}_nfft{NFFT}_bins{n_bins}.pkl\"\"\"
    feat_file  = os.path.join(output_dir, f'combined_features2_{dataset_type}_nfft{NFFT}_bins{n_bins}.pkl')
    label_file = os.path.join(output_dir, f'combined_labels2_{dataset_type}_nfft{NFFT}_bins{n_bins}.pkl')

    if os.path.exists(feat_file) and os.path.exists(label_file):
        print(f"  Loading {dataset_type} from .pkl ...")
        return joblib.load(feat_file), joblib.load(label_file)

    print(f"  Extracting {dataset_type} ...")
    t0 = time.time()
    features, labels = [], []
    for label in ['NonQueen', 'Queen']:   # Note: order is NonQueen, Queen here
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
                                                NFFT=NFFT, fmin=fmin, n_bins=n_bins,
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

# Cell 39: FFTs+CQT — Load + merge + scale
cells.append(code("""\
# Load combined FFTs+CQT features (NFFT=512, n_bins=96)
combined_train, combined_train_labels = load_combined_features_from_directory(
    train_path, NFFT=512, n_bins=96, output_dir=output_dir, dataset_type='train')
combined_val, combined_val_labels = load_combined_features_from_directory(
    val_path, NFFT=512, n_bins=96, output_dir=output_dir, dataset_type='val')
combined_test, combined_test_labels = load_combined_features_from_directory(
    test_path, NFFT=512, n_bins=96, output_dir=output_dir, dataset_type='test')

combined_train = np.vstack((combined_train, combined_val))
combined_train_labels = np.hstack((combined_train_labels, combined_val_labels))

scaler_combined = StandardScaler()
combined_train_scaled = scaler_combined.fit_transform(combined_train)
combined_test_scaled  = scaler_combined.transform(combined_test)

print(f"FFTs+CQT — Train: {combined_train.shape}, Test: {combined_test.shape}")\
"""))

# Cell 40: FFTs+CQT — Train models
cells.append(code("""\
print("=== FFTs + CQT ===")

# KNN raw (k=1) — dataset1 uses k=1
train_and_evaluate(KNeighborsClassifier(n_neighbors=1),
                   combined_train, combined_train_labels, combined_test, combined_test_labels,
                   "KNN raw (k=1)", "FFTs+CQT")

# KNN scaled (k=1)
train_and_evaluate(KNeighborsClassifier(n_neighbors=1),
                   combined_train_scaled, combined_train_labels, combined_test_scaled, combined_test_labels,
                   "KNN scaled (k=1)", "FFTs+CQT")

# SVM raw (C=100, γ=0.1)
train_and_evaluate(SVC(C=100, kernel='rbf', gamma=0.1, random_state=42),
                   combined_train, combined_train_labels, combined_test, combined_test_labels,
                   "SVM raw (C=100, γ=0.1)", "FFTs+CQT")

# SVM scaled (C=100, γ=0.1)
train_and_evaluate(SVC(C=100, kernel='rbf', gamma=0.1, random_state=42),
                   combined_train_scaled, combined_train_labels, combined_test_scaled, combined_test_labels,
                   "SVM scaled (C=100, γ=0.1)", "FFTs+CQT")

# LR raw (C=10, iter=1500) — dataset1 only
train_and_evaluate(LogisticRegression(C=10, max_iter=1500, solver='lbfgs', random_state=42),
                   combined_train, combined_train_labels, combined_test, combined_test_labels,
                   "LR raw (C=10, iter=1500)", "FFTs+CQT")

# LR scaled (C=10, iter=1000, solver=saga) — dataset1 only
train_and_evaluate(LogisticRegression(C=10, max_iter=1000, solver='saga', random_state=42),
                   combined_train_scaled, combined_train_labels, combined_test_scaled, combined_test_labels,
                   "LR scaled (C=10, saga)", "FFTs+CQT")

# RF raw
train_and_evaluate(RandomForestClassifier(n_estimators=100, random_state=42),
                   combined_train, combined_train_labels, combined_test, combined_test_labels,
                   "RF raw (n=100)", "FFTs+CQT")

# ET raw (n=200, d=30)
train_and_evaluate(ExtraTreesClassifier(n_estimators=200, max_depth=30, random_state=42),
                   combined_train, combined_train_labels, combined_test, combined_test_labels,
                   "ET raw (n=200, d=30)", "FFTs+CQT")

# ET scaled (n=200, d=30)
train_and_evaluate(ExtraTreesClassifier(n_estimators=200, max_depth=30, random_state=42),
                   combined_train_scaled, combined_train_labels, combined_test_scaled, combined_test_labels,
                   "ET scaled (n=200, d=30)", "FFTs+CQT")\
"""))

# ═══════════════════════════════════════════════════════════
# SECTION 10: PCA (FFTs + CQT)
# ═══════════════════════════════════════════════════════════
cells.append(md("# 10. PCA (FFTs + CQT)"))

cells.append(code("""\
# --- Separate FFT and CQT from the combined features, then apply PCA ---

def separate_fft_cqt(combined_features, nfft=512, n_bins=96):
    \"\"\"Split combined feature vector into FFT and CQT parts.\"\"\"
    fft_dim = nfft // 2 + 1   # 257 for NFFT=512
    if combined_features.ndim == 1:
        return combined_features[:fft_dim], combined_features[fft_dim:]
    return combined_features[:, :fft_dim], combined_features[:, fft_dim:]

# Separate into FFT (257) and CQT (96) parts
pca_train_fft, pca_train_cqt = separate_fft_cqt(combined_train, nfft=512)
pca_test_fft,  pca_test_cqt  = separate_fft_cqt(combined_test,  nfft=512)

print(f"FFT part: {pca_train_fft.shape}, CQT part: {pca_train_cqt.shape}")

# PCA on FFT: 130 components (random_state=53940)
pca_fft = PCA(n_components=130, random_state=53940)
pca_train_fft_reduced = pca_fft.fit_transform(pca_train_fft)
pca_test_fft_reduced  = pca_fft.transform(pca_test_fft)

# PCA on CQT: 80 components (random_state=53940) — dataset1 applies PCA on CQT too
pca_cqt = PCA(n_components=80, random_state=53940)
pca_train_cqt_reduced = pca_cqt.fit_transform(pca_train_cqt)
pca_test_cqt_reduced  = pca_cqt.transform(pca_test_cqt)

# Recombine: 130 + 80 = 210 features
pca_train_combined = np.hstack((pca_train_fft_reduced, pca_train_cqt_reduced))
pca_test_combined  = np.hstack((pca_test_fft_reduced,  pca_test_cqt_reduced))

print(f"PCA combined: Train {pca_train_combined.shape}, Test {pca_test_combined.shape}")\
"""))

# Cell 43: PCA (FFTs+CQT) — Train models
cells.append(code("""\
print("=== PCA (FFTs + CQT) — 210 features ===")

# SVM (C=100, γ=0.1, cache=500) — using PCA features
train_and_evaluate(SVC(C=100, kernel='rbf', gamma=0.1, cache_size=500, random_state=42),
                   pca_train_combined, combined_train_labels,
                   pca_test_combined,  combined_test_labels,
                   "SVM (C=100, γ=0.1)", "PCA(FFTs+CQT)")\
"""))

# ═══════════════════════════════════════════════════════════
# SECTION 11: PCA (FFTs + CQTs + STFT)
# ═══════════════════════════════════════════════════════════
cells.append(md("# 11. PCA (FFTs + CQTs + STFT)"))

cells.append(code("""\
# --- Re-load STFT and apply PCA, then combine with PCA(FFTs+CQT) ---

# Re-load STFT features (may already be in memory, but re-load for completeness)
pca_stft_train, _ = load_stft_features_from_directory(train_path, n_fft=4096, hop_length=1024,
                                                      output_dir=output_dir, dataset_type='train')
pca_stft_val, _   = load_stft_features_from_directory(val_path,   n_fft=4096, hop_length=1024,
                                                      output_dir=output_dir, dataset_type='val')
pca_stft_test, _  = load_stft_features_from_directory(test_path,  n_fft=4096, hop_length=1024,
                                                      output_dir=output_dir, dataset_type='test')

pca_stft_train = np.vstack((pca_stft_train, pca_stft_val))

# PCA on STFT: 80 components (random_state=3951)
pca_stft = PCA(n_components=80, random_state=3951)
pca_stft_train_reduced = pca_stft.fit_transform(pca_stft_train)
pca_stft_test_reduced  = pca_stft.transform(pca_stft_test)

print(f"STFT PCA: Train {pca_stft_train_reduced.shape}, Test {pca_stft_test_reduced.shape}")\
"""))

# Cell 45 continued: Combine everything
cells.append(code("""\
# Combine: PCA(FFTs+CQT) [210] + PCA(STFT) [80] = 290 features
final_train = np.hstack((pca_train_combined, pca_stft_train_reduced))
final_test  = np.hstack((pca_test_combined,  pca_stft_test_reduced))

print(f"Final combined: Train {final_train.shape}, Test {final_test.shape}")\
"""))

# Cell 46: PCA final — Train models
cells.append(code("""\
print("=== PCA (FFTs + CQTs + STFT) — 290 features ===")

# SVM (C=10, γ=0.0075, cache=500) — dataset1 specific params
train_and_evaluate(SVC(C=10, kernel='rbf', gamma=0.0075, cache_size=500, random_state=42),
                   final_train, combined_train_labels,
                   final_test,  combined_test_labels,
                   "SVM (C=10, γ=0.0075)", "PCA Final")

# RF (n=100, n_jobs=-1) + feature selection
print("\\n--- RF with feature selection ---")
start = time.time()
rf_final = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
rf_final.fit(final_train, combined_train_labels)
rf_preds = rf_final.predict(final_test)
rf_acc = accuracy_score(combined_test_labels, rf_preds)
print(f"  RF (n=100)                       | Acc: {rf_acc*100:.2f}% | Time: {time.time()-start:.2f}s")

# RF feature selection: top 82 features
importances = rf_final.feature_importances_
top_idx = np.argsort(importances)[::-1][:82]
rf_final.fit(final_train[:, top_idx], combined_train_labels)
rf_sel_preds = rf_final.predict(final_test[:, top_idx])
rf_sel_acc = accuracy_score(combined_test_labels, rf_sel_preds)
print(f"  RF (top 82 features)             | Acc: {rf_sel_acc*100:.2f}%")

# ET (n=200, criterion='log_loss') — dataset1 specific: uses log_loss!
train_and_evaluate(ExtraTreesClassifier(n_estimators=200, criterion='log_loss',
                                        n_jobs=-1, random_state=42),
                   final_train, combined_train_labels,
                   final_test,  combined_test_labels,
                   "ET (n=200, log_loss)", "PCA Final")\
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

output_path = r"d:\Queenless\Machine_Learning\dataset1_v2.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Created {output_path}")
print(f"Total cells: {len(cells)}")

# Count by type
md_count = sum(1 for c in cells if c['cell_type'] == 'markdown')
code_count = sum(1 for c in cells if c['cell_type'] == 'code')
print(f"  Markdown: {md_count}")
print(f"  Code: {code_count}")
