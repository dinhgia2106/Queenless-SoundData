# Kế Hoạch Refactor: dataset1.ipynb & dataset2.ipynb

## Mục tiêu

- **Đồng bộ cấu trúc** 2 notebook để dễ đọc, dễ so sánh (cùng thứ tự sections, cùng naming)
- **Clean code**: loại bỏ code trùng lặp **trong cùng 1 file**, chuẩn hóa naming, tổ chức lại
- **Kết quả giữ nguyên 100%**: KHÔNG thay đổi bất kỳ hyperparameter, logic tính toán, hay function implementation nào
- **File gốc giữ nguyên**: tạo 2 file mới `dataset1_v2.ipynb` và `dataset2_v2.ipynb`

---

## ⚠️ Nguyên tắc quan trọng: KHÔNG gộp chung chỗ nhạy cảm

Mỗi dataset dùng **bộ tham số riêng đã được tinh chỉnh**. Các phần sau **PHẢI giữ riêng biệt hoàn toàn** giữa 2 file:

### Những gì KHÁC NHAU và KHÔNG ĐƯỢC gộp

| Vùng nhạy cảm | Dataset 1 | Dataset 2 | Lý do |
|----------------|-----------|-----------|-------|
| **FFT Origin `n_fft`** | 512 | 2048 | Feature dimension hoàn toàn khác (256 vs 1024) |
| **FFTs `fft_frames()`** | `np.fft.fft(frames, NFFT)[:, :NFFT//2+1]` | `np.fft.rfft(frames, NFFT)` | Hàm FFT khác nhau |
| **FFTs log clamping** | `np.log(np.maximum(fft_feature, 1e-8))` | `np.log(fft_feature + 1e-8)` | Cách xử lý log 0 khác nhau |
| **FFTs `framing()` impl** | `np.pad()` + `np.arange` + `float()` cast | `np.zeros()` + `np.append()` + `np.tile()` | Implementation khác nhau |
| **STFTs `compute_stfts_features()`** | Gọi modular functions | Inline tất cả operations | Kiến trúc code khác |
| **STFTs SVM** | C=100, gamma=0.1 | C=80, gamma=0.08 | Params tinh chỉnh riêng |
| **STFT KNN** | n_neighbors=3 | n_neighbors=25 | Params tinh chỉnh riêng |
| **FFTs+STFT scaling** | Scale riêng từng loại → concat | Concat raw → scale chung | Logic preprocessing khác |
| **FFTs+STFT ET** | n_estimators=100 | n_estimators=200, max_depth=30 | Params khác |
| **FFTs+CQT `framing()`** | `(signal, sr, frame_size, frame_stride)` | `(signal, sr, frame_size, hop_length, overlap)` | Signature hoàn toàn khác |
| **FFTs+CQT `windowing()`** | Hamming cố định | Configurable (bartlett default) | Logic khác |
| **FFTs+CQT `compute_features()`** | 6 params | 9 params (+hop_length, overlap, window_type, normalize) | Signature khác |
| **FFTs+CQT NFFT** | 512 → feature dim 353 | 2048 → feature dim 1121 | Kích thước khác |
| **FFTs+CQT SVM** | C=100, gamma=0.1 | C=71.19, gamma=0.0375, tol=0.000532, cache_size=500 | Bayesian-optimized |
| **FFTs+CQT KNN raw** | n_neighbors=1 | n_neighbors=5 | Params khác |
| **PCA FFT n_components** | 130 | 150 | Đã tinh chỉnh riêng |
| **PCA CQT** | n_components=80 | Không apply PCA (giữ raw 96) | Logic khác hoàn toàn |
| **PCA final SVM** | C=10, gamma=0.0075 | C=71.19, gamma=0.0375, tol=0.000532 | Params khác |
| **PCA final ET** | criterion='log_loss' | Không có criterion | Params khác |
| **PKL prefixes** | ffts3, stfts, cqts, combined_features2 | fft3, stfts3, cqts3, combined_features | Tên file khác |

### Những gì AN TOÀN để đồng bộ

Chỉ đồng bộ (cùng code giữa 2 file) những phần sau:

| Phần | Nội dung | Ghi chú |
|------|----------|---------|
| **Imports** | Danh sách import libraries | Giống nhau |
| **`pre_emphasis()`** | `pre_emph=0.97` | Giống nhau, define 1 lần trong mỗi file |
| **`train_and_evaluate()` helper** | Pattern: fit → predict → accuracy → time | Chỉ format output, không ảnh hưởng logic |
| **`compute_stft_features()`** | `librosa.stft` logic | Giống nhau |
| **`compute_cqt_features()`** (standalone CQT) | `librosa.cqt` logic | Giống nhau |
| **Naming convention** | Tên biến, cấu trúc cell | Chuẩn hóa để dễ so sánh |
| **Section order** | Thứ tự các feature sections | Chuẩn hóa để dễ so sánh |
| **Markdown headers** | Tiêu đề sections | Chuẩn hóa để dễ so sánh |

---

## Phân tích chi tiết: So sánh từng section

### Models được train trong mỗi section

| Section | Model | Dataset 1 | Dataset 2 |
|---------|-------|:---------:|:---------:|
| **FFT Origin** | KNN raw | ✅ k=55 | ✅ k=19 |
| | KNN scaled | ✅ k=55 | ✅ k=19 |
| | SVM raw | ✅ C=100, γ=0.1 | ✅ C=100, γ=0.1 |
| | SVM scaled | ✅ C=100, γ=0.1 | ✅ C=100, γ=0.1 |
| | RF raw | ✅ n=100 | ✅ n=100 |
| | ET raw | ✅ n=200, d=30 | ✅ n=200, d=30 |
| | ET scaled | ✅ n=200, d=30 | ❌ |
| **FFTs** | KNN | ✅ k=5 | ✅ k=5 |
| | SVM | ✅ C=100, γ=0.1 | ✅ C=100, γ=0.1 |
| | RF | ✅ n=100 | ✅ n=100 |
| | ET raw | ✅ n=200, d=30 | ✅ n=200, d=30 |
| | ET scaled | ✅ | ❌ |
| **STFT** | KNN | ✅ k=3 | ✅ k=25 |
| | SVM | ✅ C=10, γ=scale | ✅ C=10, γ=scale |
| | RF | ❌ | ✅ n=100 |
| | ET | ❌ | ✅ n=100 |
| **STFTs** | KNN | ✅ k=5 | ✅ k=5 |
| | SVM | ✅ C=100, γ=0.1 | ✅ C=80, γ=0.08 |
| | RF | ✅ n=100 | ✅ n=100 |
| | ET | ✅ n=200, d=30 | ✅ n=200, d=30 |
| **CQT** | KNN scaled | ✅ k=1 | ✅ k=1 |
| | SVM raw | ❌ | ✅ C=100, γ=auto |
| | SVM scaled | ✅ C=100, γ=auto | ✅ C=100, γ=auto |
| | LR | ✅ C=1, iter=500 | ✅ C=1, iter=500 |
| | RF | ✅ n=100, bootstrap=F | ✅ n=100, bootstrap=F |
| | ET scaled | ✅ n=200, d=30 | ✅ n=200, d=30 |
| **FFTs+CQT** | KNN raw | ✅ k=1 | ✅ k=5 |
| | KNN scaled | ✅ k=1 | ✅ k=1 |
| | SVM | ✅ C=100, γ=0.1 | ✅ C=71.19, γ=0.0375 (tuned) |
| | LR | ✅ C=10 | ❌ |
| | RF | ✅ n=100 | ✅ n=100 |
| | ET | ✅ n=200, d=30 | ✅ n=200, d=30 |

---

## Cấu trúc mới (chuẩn hóa cho cả 2 notebook)

> Cả 2 file cùng thứ tự sections, cùng markdown headers, cùng naming.
> Nhưng **nội dung code bên trong** mỗi cell giữ nguyên logic/params riêng của từng dataset.

```
Cell 1:  [MD]   # Title & Description
Cell 2:  [Code] Imports (GIỐNG NHAU)
Cell 3:  [Code] Paths & Labels (KHÁC NHAU - dataset-specific)
Cell 4:  [Code] train_and_evaluate() helper (GIỐNG NHAU)

--- Section 1: FFT Origin ---
Cell 5:  [MD]   # 1. FFT (Origin)
Cell 6:  [Code] extract_fft_features() + load_dataset_fft()  ← KHÁC n_fft, labels
Cell 7:  [Code] Load data + Scale
Cell 8:  [Code] Train models  ← KHÁC hyperparams, KHÁC models trained

--- Section 2: FFTs (Pre-emphasis + Framing + Windowing + FFT) ---
Cell 9:  [MD]   # 2. FFTs
Cell 10: [Code] pre_emphasis() + framing() + windowing() + fft_frames()  ← KHÁC implementation
Cell 11: [Code] compute_fft_features() + load_from_directory()  ← KHÁC log clamping, pkl prefix
Cell 12: [Code] Load data + merge train+val + Scale
Cell 13: [Code] Train models  ← KHÁC models trained

--- Section 3: STFT ---
Cell 14: [MD]   # 3. STFT
Cell 15: [Code] compute_stft_features() + load_from_directory()  ← GIỐNG logic
Cell 16: [Code] Load data + merge + Scale
Cell 17: [Code] Train models  ← KHÁC k, KHÁC models (ds1 no RF/ET, ds2 has RF/ET)

--- Section 4: FFTs + STFT (Combine) ---
Cell 18: [MD]   # 4. FFTs + STFT
Cell 19: [Code] Combine + Scale  ← KHÁC scaling approach!
Cell 20: [Code] Train models  ← KHÁC ET params

--- Section 5: STFTs (Pre-emphasis + STFT per frame) ---
Cell 21: [MD]   # 5. STFTs
Cell 22: [Code] compute_stfts_features() + load_from_directory()  ← KHÁC implementation (modular vs inline)
Cell 23: [Code] Load data + merge + Scale
Cell 24: [Code] Train models  ← KHÁC SVM params

--- Section 6: FFTs + STFTs (Combine) ---
Cell 25: [MD]   # 6. FFTs + STFTs
Cell 26: [Code] Combine + Scale
Cell 27: [Code] Train models

--- Section 7: CQT ---
Cell 28: [MD]   # 7. CQT
Cell 29: [Code] compute_cqt_features() + load_from_directory()  ← GIỐNG logic
Cell 30: [Code] Load data + merge + Scale
Cell 31: [Code] Train models  ← KHÁC models (ds2 has extra raw SVM)

--- Section 8: CQTs (Pre-emphasis + CQT) ---
Cell 32: [MD]   # 8. CQTs
Cell 33: [Code] compute_cqts_features() + load_from_directory()  ← GIỐNG logic, KHÁC pkl prefix
Cell 34: [Code] Load data + merge + Scale
Cell 35: [Code] Train models

--- Section 9: FFTs + CQT (Combined extraction) ---
Cell 36: [MD]   # 9. FFTs + CQT
Cell 37: [Code] framing() + windowing() + compute_features()  ← KHÁC HOÀN TOÀN (signature, logic)
Cell 38: [Code] load_features_from_directory()  ← KHÁC HOÀN TOÀN (signature, pkl naming)
Cell 39: [Code] Load data + merge + Scale
Cell 40: [Code] Train models  ← KHÁC SVM (tuned), KHÁC KNN, ds1 has LR

--- Section 10: PCA (FFTs + CQT) ---
Cell 41: [MD]   # 10. PCA (FFTs + CQT)
Cell 42: [Code] Separate FFT/CQT + Apply PCA  ← KHÁC n_components, ds2 no CQT PCA
Cell 43: [Code] Train models

--- Section 11: PCA (FFTs + CQTs + STFT) ---
Cell 44: [MD]   # 11. PCA (FFTs + CQTs + STFT)
Cell 45: [Code] Load STFT + PCA + Combine  ← KHÁC combined shape
Cell 46: [Code] Train models  ← KHÁC SVM, ET params
```

---

## Phases Triển Khai

---

### Phase 1: Imports + Paths + Helper (Cell 1–4)

**Mục tiêu**: Tạo phần đầu notebook chuẩn. Chỉ đồng bộ những gì AN TOÀN.

**Thay đổi:**

1. **Cell 1 - Markdown Title** (KHÁC NHAU)
   - Dataset1: `# Machine Learning Pipeline - Dataset 1 (20k Audio)`
   - Dataset2: `# Machine Learning Pipeline - Dataset 2 (NuHive/Kaggle)`

2. **Cell 2 - Imports** (GIỐNG NHAU)
   ```python
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
   ```

3. **Cell 3 - Paths & Labels** (KHÁC NHAU - chỉ paths/labels, KHÔNG gom params)

   **Dataset 1:**
   ```python
   # ============ DATASET PATHS ============
   DATASET_NAME = "Dataset 1 - 20k Audio"
   BASE_PATH = r"D:\Queenless\20k_audio_splitted_dataset"
   FEATURE_DIR = r"D:\Queenless\features"
   LABELS = ['Queen', 'NonQueen']

   TRAIN_DIR = os.path.join(BASE_PATH, 'train')
   VAL_DIR = os.path.join(BASE_PATH, 'val')
   TEST_DIR = os.path.join(BASE_PATH, 'test')
   os.makedirs(FEATURE_DIR, exist_ok=True)
   ```

   **Dataset 2:**
   ```python
   # ============ DATASET PATHS ============
   DATASET_NAME = "Dataset 2 - NuHive/Kaggle"
   BASE_PATH = r"D:\Queenless\archive\nuhive_processed"
   FEATURE_DIR = r"D:\Queenless\kaggle_features"
   LABELS = ['bee', 'nobee', 'noqueen']

   TRAIN_DIR = os.path.join(BASE_PATH, 'train')
   VAL_DIR = os.path.join(BASE_PATH, 'val')
   TEST_DIR = os.path.join(BASE_PATH, 'test')
   os.makedirs(FEATURE_DIR, exist_ok=True)
   ```

   > **Không gom hyperparams vào config!** Mỗi section giữ params inline để:
   > - Dễ thấy params đang dùng khi đọc section đó
   > - Không bị nhầm lẫn giữa 2 file
   > - Không bị sửa nhầm config ảnh hưởng nhiều sections

4. **Cell 4 - Train & Evaluate Helper** (GIỐNG NHAU)
   ```python
   def train_and_evaluate(clf, X_train, y_train, X_test, y_test, model_name, feature_name):
       """Train model, predict, print accuracy & time."""
       start = time.time()
       clf.fit(X_train, y_train)
       train_time = time.time() - start
       preds = clf.predict(X_test)
       acc = accuracy_score(y_test, preds)
       print(f"  {model_name:20s} | Accuracy: {acc*100:.2f}% | Time: {train_time:.2f}s")
       return acc, preds
   ```
   > Hàm này CHỈ wrap pattern fit→predict→print. KHÔNG quyết định model nào được train.
   > Mỗi section tự chọn models + hyperparams riêng.

**Checklist Phase 1:**
- [ ] Cell 1: Markdown title khác nhau
- [ ] Cell 2: Imports giống nhau, chạy OK
- [ ] Cell 3: Chỉ có paths + labels, không có hyperparams
- [ ] Cell 4: `train_and_evaluate()` helper, không có `evaluate_all_models()`
- [ ] Verify: Không cell nào dưới import lại thư viện

---

### Phase 2: FFT Origin (Cell 5–8)

**Mục tiêu**: Chuẩn hóa section FFT gốc (không pre-emphasis). Giữ nguyên params riêng.

**Cell 6 - Functions** (KHÁC NHAU)

| | Dataset 1 | Dataset 2 |
|--|-----------|-----------|
| `n_fft` | **512** → 256 features | **2048** → 1024 features |
| `extract_fft_features()` | Dùng `n_fft=512` | Dùng `n_fft=2048` |
| `load_dataset_fft()` | Dùng `LABELS` | Dùng `LABELS` |

**Cell 8 - Train models** (KHÁC NHAU)

| Model | Dataset 1 | Dataset 2 |
|-------|-----------|-----------|
| KNN raw | k=**55** | k=**19** |
| KNN scaled | k=**55** | k=**19** |
| SVM raw | C=100, γ=0.1 | C=100, γ=0.1 |
| SVM scaled | C=100, γ=0.1 | C=100, γ=0.1 |
| RF raw | n=100 | n=100 |
| ET raw | n=200, d=30 | n=200, d=30 |
| ET scaled | ✅ **CÓ train** | ❌ **KHÔNG train** |

> Model code KHÔNG dùng shared function. Mỗi file tự instantiate models với params riêng.

**Checklist Phase 2:**
- [ ] Cell 5: Header `# 1. FFT (Origin)`
- [ ] Cell 6: `extract_fft_features()` giữ nguyên n_fft riêng
- [ ] Cell 7: Load + merge + scale
- [ ] Cell 8: Train models giữ nguyên params + giữ/bỏ ET scaled đúng theo file gốc
- [ ] Verify accuracy match file gốc

---

### Phase 3: FFTs - Pre-emphasis Pipeline (Cell 9–13)

**Mục tiêu**: Chuẩn hóa section FFTs. ⚠️ **NHẠY CẢM** - implementation khác nhau.

**Cell 10 - Signal Processing** (KHÁC NHAU - mỗi file giữ implementation riêng)

| Hàm | Dataset 1 | Dataset 2 |
|-----|-----------|-----------|
| `pre_emphasis()` | Giống | Giống |
| `framing()` | `np.pad()` + `float(np.abs(...))` | `np.zeros()` + `np.append()` |
| `windowing()` | `np.hamming()` | `np.hamming()` |
| `fft_frames()` | **`np.fft.fft(frames, NFFT)[:, :NFFT//2+1]`** | **`np.fft.rfft(frames, NFFT)`** |

> Tuy kết quả "gần tương đương" nhưng **KHÔNG chắc chắn giống bit-by-bit**.
> **Giữ nguyên implementation gốc** của từng file → kết quả y hệt.

**Cell 11 - Compute + Load** (KHÁC NHAU)

| | Dataset 1 | Dataset 2 |
|--|-----------|-----------|
| Log | `np.log(np.maximum(fft_feature, 1e-8))` | `np.log(fft_feature + 1e-8)` |
| PKL prefix | `ffts3_` | `fft3_` |

**Checklist Phase 3:**
- [ ] Cell 9: Header `# 2. FFTs`
- [ ] Cell 10: `framing()`, `fft_frames()` giữ implementation riêng, KHÔNG copy từ file kia
- [ ] Cell 11: Log clamping giữ riêng, PKL prefix giữ riêng
- [ ] Cell 12: Load + merge + scale
- [ ] Cell 13: Train models, verify accuracy

---

### Phase 4: STFT (Cell 14–17)

**Mục tiêu**: Chuẩn hóa section STFT. Phần này tương đối GIỐNG logic trích xuất.

**Cell 15 - Functions** (GIỐNG logic, chỉ khác tham số gọi)

| | Dataset 1 | Dataset 2 |
|--|-----------|-----------|
| `compute_stft_features()` | Giống | Giống |
| `n_fft` | 4096 | 4096 |
| `hop_length` | 1024 | 1024 |
| PKL prefix | `stft2_` | `stft2_` |

**Cell 17 - Train models** (KHÁC NHAU)

| Model | Dataset 1 | Dataset 2 |
|-------|-----------|-----------|
| KNN | k=**3** | k=**25** |
| SVM | C=10, γ=scale | C=10, γ=scale |
| RF | ❌ **KHÔNG train** | ✅ n=100 |
| ET | ❌ **KHÔNG train** | ✅ n=100 |

**Checklist Phase 4:**
- [ ] Cell 14: Header `# 3. STFT`
- [ ] Cell 15: Functions có thể giống nhau (an toàn)
- [ ] Cell 16: Load + merge + scale
- [ ] Cell 17: Train models giữ riêng (ds1 no RF/ET)
- [ ] Verify accuracy

---

### Phase 5: FFTs + STFT Combination (Cell 18–20)

**Mục tiêu**: Combine FFTs + STFT features. ⚠️ **NHẠY CẢM** - scaling approach khác.

**Cell 19 - Combine + Scale** (KHÁC NHAU)

| | Dataset 1 | Dataset 2 |
|--|-----------|-----------|
| Scaling | **Scale riêng FFTs, riêng STFT → concat** | **Concat raw → scale chung** |

> Đây là khác biệt về logic preprocessing, KHÔNG được thay đổi.

**Cell 20 - Train models** (KHÁC NHAU)

| Model | Dataset 1 | Dataset 2 |
|-------|-----------|-----------|
| KNN | k=**3** | k=**5** |
| ET | n=**100** | n=**200**, d=**30** |

**Checklist Phase 5:**
- [ ] Cell 18: Header `# 4. FFTs + STFT`
- [ ] Cell 19: Scaling approach giữ riêng
- [ ] Cell 20: Train models giữ riêng
- [ ] Verify accuracy

---

### Phase 6: STFTs - Pre-emphasis + STFT per frame (Cell 21–24)

**Mục tiêu**: Chuẩn hóa section STFTs. ⚠️ **NHẠY CẢM** - implementation khác.

**Cell 22 - Functions** (KHÁC NHAU)

| | Dataset 1 | Dataset 2 |
|--|-----------|-----------|
| `compute_stfts_features()` | **Gọi modular**: `pre_emphasis()` → `framing()` → `windowing()` | **Inline tất cả**: code pre_emphasis/padding/windowing ngay trong hàm |
| `load_stfts` safeguards | Basic | Extra: `os.makedirs()`, `.wav` check, `os.path.isdir()` |
| PKL prefix | `stfts_` | `stfts3_` |

**Cell 24 - Train models** (KHÁC NHAU)

| Model | Dataset 1 | Dataset 2 |
|-------|-----------|-----------|
| SVM | C=**100**, γ=**0.1** | C=**80**, γ=**0.08** |
| SVM scaled | ✅ **CÓ** | ❌ **KHÔNG** |

**Checklist Phase 6:**
- [ ] Cell 21: Header `# 5. STFTs`
- [ ] Cell 22: Giữ implementation riêng (modular vs inline)
- [ ] Cell 23: Load + merge + scale
- [ ] Cell 24: SVM params giữ riêng
- [ ] Verify accuracy

---

### Phase 7: FFTs + STFTs Combination (Cell 25–27)

**Mục tiêu**: Combine FFTs + STFTs features.

**Phần này tương đối giống** (cùng params KNN/SVM/RF/ET).

**Checklist Phase 7:**
- [ ] Cell 25: Header `# 6. FFTs + STFTs`
- [ ] Combine + scale + train
- [ ] Verify accuracy

---

### Phase 8: CQT (Cell 28–31)

**Mục tiêu**: Chuẩn hóa section CQT. Logic trích xuất GIỐNG, models KHÁC.

**Cell 29** (GIỐNG logic)
- `compute_cqt_features()`: giống cả 2
- `n_bins=84`, `bins_per_octave=12`, `hop_length=512`

**Cell 31 - Train models** (KHÁC NHAU)

| Model | Dataset 1 | Dataset 2 |
|-------|-----------|-----------|
| SVM raw | ❌ | ✅ C=100, γ=auto |
| SVM scaled | ✅ | ✅ |
| Còn lại | Giống | Giống |

**Checklist Phase 8:**
- [ ] Cell 28: Header `# 7. CQT`
- [ ] Cell 29: Functions giống nhau (an toàn)
- [ ] Cell 31: Ds2 thêm raw SVM
- [ ] Verify accuracy

---

### Phase 9: CQTs - Pre-emphasis + CQT (Cell 32–35)

**Mục tiêu**: Chuẩn hóa section CQTs.

**Cell 33** (KHÁC pkl prefix + safeguards)

| | Dataset 1 | Dataset 2 |
|--|-----------|-----------|
| PKL prefix | `cqts_` | `cqts3_` |
| Load safeguards | Basic | Extra checks |

**Checklist Phase 9:**
- [ ] Cell 32: Header `# 8. CQTs`
- [ ] PKL prefix giữ riêng
- [ ] Verify accuracy

---

### Phase 10: FFTs + CQT Combined (Cell 36–40) ⚠️ SECTION KHÁC NHẤT

**Mục tiêu**: Refactor chỉ về structure, **code bên trong hoàn toàn riêng**.

> **Đây là section khác nhau nhất giữa 2 file!**
> Dataset 2 có extended signatures với hop_length, overlap, window_type, normalize.
> KHÔNG được dùng chung function definitions.

**Cell 37 - Core Functions** (KHÁC HOÀN TOÀN)

| Hàm | Dataset 1 | Dataset 2 |
|-----|-----------|-----------|
| `framing()` | `(signal, sr, frame_size=0.025, frame_stride=0.01)` | `(signal, sr, frame_size=0.025, hop_length=256, overlap=0.6)` |
| `windowing()` | `(frames)` → Hamming only | `(frames, window_type='bartlett')` → configurable |
| `compute_fft_features()` | 6 params | 9 params (+hop_length, overlap, window_type, normalize) |
| `compute_cqt_features()` | No hop_length, no normalize | Has hop_length + normalize |
| `compute_features()` | Simple | Extended signature |
| `load_features()` | Simple pkl naming | Extended pkl naming (+hop in filename) |
| `NFFT` | **512** → 353 total features | **2048** → 1121 total features |

**Cell 40 - Train models** (KHÁC NHAU)

| Model | Dataset 1 | Dataset 2 |
|-------|-----------|-----------|
| KNN raw | k=**1** | k=**5** |
| KNN scaled | k=1 | k=1 |
| SVM | C=100, γ=0.1 | C=**71.19**, γ=**0.0375**, tol=**0.000532**, cache=**500** |
| LR | ✅ C=10, iter=1500 | ❌ KHÔNG CÓ |
| RF | n=100 | n=100 |
| ET | n=200, d=30 | n=200, d=30 |

**Checklist Phase 10:**
- [ ] Cell 36: Header `# 9. FFTs + CQT`
- [ ] Cell 37: Functions HOÀN TOÀN RIÊNG, copy từ file gốc
- [ ] Cell 38: Load functions HOÀN TOÀN RIÊNG
- [ ] Cell 40: Models + params riêng (Bayesian-tuned SVM cho ds2, LR chỉ ds1)
- [ ] Verify accuracy

---

### Phase 11: PCA Sections (Cell 41–46) ⚠️ NHẠY CẢM

**Mục tiêu**: Chuẩn hóa PCA pipeline. Logic khác nhau đáng kể.

**PCA (FFTs + CQT) - Cell 42** (KHÁC NHAU)

| | Dataset 1 | Dataset 2 |
|--|-----------|-----------|
| FFT split dim | NFFT//2+1 = **257** | NFFT//2+1 = **1025** |
| FFT PCA n_components | **130** | **150** |
| CQT PCA | n_components=**80** | ❌ **KHÔNG apply PCA** (giữ raw 96) |
| Combined shape | 130+80 = **210** | 150+96 = **246** |

**PCA (FFTs + CQTs + STFT) - Cell 45–46** (KHÁC NHAU)

| | Dataset 1 | Dataset 2 |
|--|-----------|-----------|
| STFT PCA | n_components=80 | n_components=80 |
| Final shape | 130+80+80 = **290** | 150+96+80 = **326** |
| SVM | C=**10**, γ=**0.0075** | C=**71.19**, γ=**0.0375**, tol=0.000532 |
| ET | criterion=**'log_loss'** | Không có criterion param |

**Checklist Phase 11:**
- [ ] Cell 41: Header `# 10. PCA (FFTs + CQT)`
- [ ] Cell 42: PCA n_components riêng, CQT PCA logic riêng
- [ ] Cell 44: Header `# 11. PCA (FFTs + CQTs + STFT)`
- [ ] Cell 46: SVM/ET params riêng
- [ ] Verify accuracy

---

## Quy tắc khi thực hiện

### 1. Naming Convention (đồng bộ cả 2 file)

| Loại | Convention | Ví dụ |
|------|-----------|-------|
| Feature arrays | `{feat}_{split}_features` | `fft_train_features`, `stft_test_features` |
| Label arrays | `{feat}_{split}_labels` | `fft_train_labels` |
| Scaled arrays | `{feat}_{split}_scaled` | `fft_train_scaled` |
| Predictions | `preds_{model}_{feat}` | `preds_knn_fft`, `preds_svm_ffts` |
| Accuracy | `acc_{model}_{feat}` | `acc_knn_fft`, `acc_svm_ffts` |

### 2. Cell Structure Pattern cho mỗi Feature Section

```
[MD]   # N. Feature Name
[Code] Define functions (extract/compute + load)    ← DATASET-SPECIFIC
[Code] Load train/val/test → merge → scale          ← Cùng pattern, khác data
[Code] Train & evaluate models                       ← DATASET-SPECIFIC params
```

### 3. Nguyên tắc "Giữ riêng"

- **Hyperparameters**: LUÔN hardcode inline trong cell train models. KHÔNG đặt trong config cell.
- **Feature extraction functions**: Nếu implementation khác → copy nguyên từ file gốc, KHÔNG sửa.
- **Scaling approach**: Giữ nguyên cách scale gốc (riêng vs chung).
- **Models trained**: Giữ nguyên list models (không thêm/bớt model nào so với gốc).

### 4. Nguyên tắc "Đồng bộ"

- **Section order**: Cùng thứ tự sections.
- **Markdown headers**: Cùng format headers.
- **Variable naming**: Cùng tên biến (theo convention trên).
- **Cell organization**: Cùng số cells per section, cùng mục đích.
- **Imports**: 1 cell duy nhất ở đầu.
- **`train_and_evaluate()` helper**: Chỉ wrap fit→predict→print, dùng chung.

---

## Thứ tự thực hiện

1. **Phase 1** → Test: imports, paths, helper chạy OK
2. **Phase 2** → Test: FFT origin accuracy matches
3. **Phase 3** → Test: FFTs accuracy matches (cẩn thận implementation)
4. **Phase 4** → Test: STFT accuracy matches
5. **Phase 5** → Test: FFTs+STFT accuracy matches (cẩn thận scaling)
6. **Phase 6** → Test: STFTs accuracy matches (cẩn thận SVM params)
7. **Phase 7** → Test: FFTs+STFTs accuracy matches
8. **Phase 8** → Test: CQT accuracy matches
9. **Phase 9** → Test: CQTs accuracy matches
10. **Phase 10** → Test: FFTs+CQT accuracy matches (**section khác nhất - cẩn thận nhất**)
11. **Phase 11** → Test: PCA results match

> **⚠️ Mỗi phase chạy xong phải verify accuracy trước khi sang phase tiếp theo!**
> **⚠️ Nếu accuracy khác → STOP, kiểm tra lại code gốc, KHÔNG tiếp tục!**
