train_path = 'E:/Queenless/20k_audio_splitted_dataset/train'
val_path = 'E:/Queenless/20k_audio_splitted_dataset/val'
test_path = 'E:/Queenless/20k_audio_splitted_dataset/test'
import os
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import librosa.display
import scipy.fftpack as fftpack
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

def pre_emphasis(signal_in, alpha=0.97):
    """
    Bước 1: Pre-emphasis - Lọc thông cao
    """
    emphasized_signal = np.append(signal_in[0], signal_in[1:] - alpha * signal_in[:-1]) # y(t) = x(t) - alpha*x(t-1)
    return emphasized_signal

def framing(signal_in, sample_rate, frame_size=0.025, frame_stride=0.01):
    """
    Bước 2: Chia khung (Framing)
    - frame_size: kích thước khung (số giây)
    - frame_stride: bước nhảy giữa các khung (số giây)
    """
    frame_length = int(round(frame_size * sample_rate))
    frame_step = int(round(frame_stride * sample_rate))
    signal_length = len(signal_in)
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step)) + 1

    pad_signal_length = num_frames * frame_step + frame_length
    # Zero-padding nếu cần
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(signal_in, z)

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    return frames

def windowing(frames):
    """
    Bước 3: Áp dụng cửa sổ Hamming cho mỗi khung
    """
    frame_length = frames.shape[1]
    hamming = np.hamming(frame_length)
    windowed_frames = frames * hamming
    return windowed_frames

def fft_frames(frames, NFFT=512):
    """
    Bước 4: Tính FFT cho mỗi khung
    """
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    return mag_frames

def power_spectrum(mag_frames, NFFT=512):
    """
    Bước 4.1: Tính phổ công suất của mỗi khung
    """
    return (1.0 / NFFT) * (mag_frames ** 2)

def mel_filterbank(sample_rate, NFFT, nfilt=26, low_freq=0, high_freq=None):
    """
    Bước 5: Tạo Mel filterbank
    """
    if high_freq is None:
        high_freq = sample_rate / 2

    # Chuyển Hz sang Mel
    low_mel = 2595 * np.log10(1 + low_freq / 700.0)
    high_mel = 2595 * np.log10(1 + high_freq / 700.0)
    mel_points = np.linspace(low_mel, high_mel, nfilt + 2)
    # Chuyển lại từ Mel sang Hz
    hz_points = 700 * (10**(mel_points / 2595) - 1)
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # giới hạn trái
        f_m = int(bin[m])             # trung tâm
        f_m_plus = int(bin[m + 1])    # giới hạn phải

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    return fbank

# Hàm tính MFCC
def compute_mfcc(signal_in, sample_rate, frame_size=0.025, frame_stride=0.01, 
                 pre_emph=0.97, NFFT=512, nfilt=26, num_ceps=13):
    emphasized_signal = pre_emphasis(signal_in, pre_emph)
    frames = framing(emphasized_signal, sample_rate, frame_size, frame_stride)
    windowed_frames = windowing(frames)
    mag_frames = fft_frames(windowed_frames, NFFT)
    pow_frames = power_spectrum(mag_frames, NFFT)
    fbank = mel_filterbank(sample_rate, NFFT, nfilt)
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    log_fbank = np.log(filter_banks)
    mfccs = fftpack.dct(log_fbank, type=2, axis=1, norm='ortho')[:, :num_ceps]
    return mfccs
def load_data_from_directory(directory, sample_rate=22050):
    labels = []
    features = []
    for label in ['Queen', 'NonQueen']:
        path = os.path.join(directory, label)
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            signal, sr = librosa.load(file_path, sr=sample_rate)
            mfcc = compute_mfcc(signal_in=signal, sample_rate=sr, frame_size=0.025, frame_stride=0.01, 
                 pre_emph=0.97, NFFT=512, nfilt=26, num_ceps=13)
            mfcc_mean = np.mean(mfcc, axis=0)  # Tính trung bình của MFCCs để giảm chiều
            features.append(mfcc_mean)
            labels.append(label)
    return np.array(features), np.array(labels)

# Load training, validation và testing data
train_features, train_labels = load_data_from_directory(train_path, sample_rate=22050)
val_features, val_labels = load_data_from_directory(val_path, sample_rate=22050)
test_features, test_labels = load_data_from_directory(test_path, sample_rate=22050)
print(train_features.shape)
print(val_features.shape)
print(test_features.shape)
train_features[0, :]
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
val_features_scaled = scaler.transform(val_features)
test_features_scaled = scaler.transform(test_features)
scaler = MinMaxScaler()
train_features_mmscaled = scaler.fit_transform(train_features)
val_features_mmscaled = scaler.transform(val_features)
test_features_mmscaled = scaler.transform(test_features)
# Huấn luyện mô hình Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, 
                                       min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
                                       max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, 
                                       bootstrap=False, oob_score=False, n_jobs=None, random_state=42, verbose=0, 
                                       warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)
rf_classifier.fit(train_features, train_labels)

# Đánh giá mô hình trên bộ validation
val_predictions = rf_classifier.predict(val_features)
val_accuracy = accuracy_score(val_labels, val_predictions)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Đánh giá mô hình trên bộ testing
test_predictions = rf_classifier.predict(test_features)
test_accuracy_mfcc_rf = accuracy_score(test_labels, test_predictions)
print(f"Test Accuracy: {test_accuracy_mfcc_rf * 100:.2f}%")

# Khởi tạo mô hình SVM với kernel RBF
svm_rbf_classifier = SVC(C=10, kernel='rbf', degree=3, gamma=1, coef0=0.0, shrinking=True, 
                         probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, 
                         max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=42)

# Huấn luyện mô hình SVM
svm_rbf_classifier.fit(train_features, train_labels)

# Đánh giá mô hình trên bộ validation
val_predictions_svm_rbf = svm_rbf_classifier.predict(val_features)
val_accuracy_svm_rbf = accuracy_score(val_labels, val_predictions_svm_rbf)
print(f"Validation Accuracy (SVM with RBF Kernel): {val_accuracy_svm_rbf * 100:.2f}%")

# Đánh giá mô hình trên bộ testing
test_predictions_svm_rbf = svm_rbf_classifier.predict(test_features)
test_accuracy_mfcc_svm = accuracy_score(test_labels, test_predictions_svm_rbf)
print(f"Test Accuracy (SVM with RBF Kernel): {test_accuracy_mfcc_svm * 100:.2f}%")
# Khởi tạo mô hình SVM với data scaling
svm_rbf_classifier = SVC(C=10, kernel='rbf', degree=3, gamma=1, coef0=0.0, shrinking=True, 
                         probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, 
                         max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=42)

# Huấn luyện mô hình SVM
svm_rbf_classifier.fit(train_features_scaled, train_labels)

# Đánh giá mô hình trên bộ validation
val_predictions_svm_rbf = svm_rbf_classifier.predict(val_features_scaled)
val_accuracy_svm_rbf = accuracy_score(val_labels, val_predictions_svm_rbf)
print(f"Validation Accuracy (SVM with RBF Kernel): {val_accuracy_svm_rbf * 100:.2f}%")

# Đánh giá mô hình trên bộ testing
test_predictions_svm_rbf = svm_rbf_classifier.predict(test_features_scaled)
scale_test_accuracy_mfcc_svm = accuracy_score(test_labels, test_predictions_svm_rbf)
print(f"Test Accuracy (SVM with RBF Kernel): {scale_test_accuracy_mfcc_svm * 100:.2f}%")
# Khởi tạo mô hình Logistic Regression
lr_classifier = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, 
                                   class_weight=None, random_state=42, solver='liblinear', max_iter=100, multi_class='deprecated', 
                                   verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)

# Huấn luyện mô hình Logistic Regression với dữ liệu đã chuẩn hóa
lr_classifier.fit(train_features, train_labels)

# Đánh giá mô hình trên bộ validation
val_predictions_lr = lr_classifier.predict(val_features)
val_accuracy_lr = accuracy_score(val_labels, val_predictions_lr)
print(f"Validation Accuracy (Logistic Regression): {val_accuracy_lr * 100:.2f}%")

# Đánh giá mô hình trên bộ testing
test_predictions_lr = lr_classifier.predict(test_features)
test_accuracy_mfcc_lr = accuracy_score(test_labels, test_predictions_lr)
print(f"Test Accuracy (Logistic Regression): {test_accuracy_mfcc_lr * 100:.2f}%")
# Khởi tạo mô hình Logistic Regression với data scaling
lr_classifier = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, 
                                   class_weight=None, random_state=42, solver='liblinear', max_iter=100, multi_class='deprecated', 
                                   verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)

# Huấn luyện mô hình Logistic Regression với dữ liệu đã chuẩn hóa
lr_classifier.fit(train_features_scaled, train_labels)

# Đánh giá mô hình trên bộ validation
val_predictions_lr = lr_classifier.predict(val_features_scaled)
val_accuracy_lr = accuracy_score(val_labels, val_predictions_lr)
print(f"Validation Accuracy (Logistic Regression): {val_accuracy_lr * 100:.2f}%")

# Đánh giá mô hình trên bộ testing
test_predictions_lr = lr_classifier.predict(test_features_scaled)
scale_test_accuracy_mfcc_lr = accuracy_score(test_labels, test_predictions_lr)
print(f"Test Accuracy (Logistic Regression): {scale_test_accuracy_mfcc_lr * 100:.2f}%")
# Khởi tạo mô hình Extra Trees 
et_classifier = ExtraTreesClassifier(n_estimators=200, criterion='gini', max_depth=30, min_samples_split=2, min_samples_leaf=1, 
                                     min_weight_fraction_leaf=0.0, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0,
                                     bootstrap=False, oob_score=False, n_jobs=None, random_state=42, verbose=0, warm_start=False, 
                                     class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)

# Huấn luyện mô hình Extra Trees với dữ liệu đã chuẩn hóa
et_classifier.fit(train_features, train_labels)

# Đánh giá mô hình trên bộ validation
val_predictions_et = et_classifier.predict(val_features)
val_accuracy_et = accuracy_score(val_labels, val_predictions_et)
print(f"Validation Accuracy (Extra Trees): {val_accuracy_et * 100:.2f}%")

# Đánh giá mô hình trên bộ testing
test_predictions_et = et_classifier.predict(test_features)
test_accuracy_mfcc_et = accuracy_score(test_labels, test_predictions_et)
print(f"Test Accuracy (Extra Trees): {test_accuracy_mfcc_et * 100:.2f}%")

# Khởi tạo mô hình Extra Trees với data scaling
et_classifier = ExtraTreesClassifier(n_estimators=200, criterion='gini', max_depth=30, min_samples_split=2, min_samples_leaf=1, 
                                     min_weight_fraction_leaf=0.0, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0,
                                     bootstrap=False, oob_score=False, n_jobs=None, random_state=42, verbose=0, warm_start=False, 
                                     class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)

# Huấn luyện mô hình Extra Trees với dữ liệu đã chuẩn hóa
et_classifier.fit(train_features_scaled, train_labels)

# Đánh giá mô hình trên bộ validation
val_predictions_et = et_classifier.predict(val_features_scaled)
val_accuracy_et = accuracy_score(val_labels, val_predictions_et)
print(f"Validation Accuracy (Extra Trees): {val_accuracy_et * 100:.2f}%")

# Đánh giá mô hình trên bộ testing
test_predictions_et = et_classifier.predict(test_features_scaled)
scale_test_accuracy_mfcc_et = accuracy_score(test_labels, test_predictions_et)
print(f"Test Accuracy (Extra Trees): {scale_test_accuracy_mfcc_et * 100:.2f}%")

knn_classifier = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, 
                                      metric='minkowski', metric_params=None, n_jobs=None)

# Huấn luyện mô hình KNN với dữ liệu đã chuẩn hóa
knn_classifier.fit(train_features, train_labels)

# Đánh giá mô hình trên bộ validation
val_predictions_knn = knn_classifier.predict(val_features)
val_accuracy_knn = accuracy_score(val_labels, val_predictions_knn)
print(f"Validation Accuracy (KNN): {val_accuracy_knn * 100:.2f}%")

# Đánh giá mô hình trên bộ testing
test_predictions_knn = knn_classifier.predict(test_features)
test_accuracy_mfcc_knn = accuracy_score(test_labels, test_predictions_knn)
print(f"Test Accuracy (KNN): {test_accuracy_mfcc_knn * 100:.2f}%")
knn_classifier = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, 
                                      metric='minkowski', metric_params=None, n_jobs=None)

# Huấn luyện mô hình KNN với dữ liệu đã chuẩn hóa
knn_classifier.fit(train_features_scaled, train_labels)

# Đánh giá mô hình trên bộ validation
val_predictions_knn = knn_classifier.predict(val_features_scaled)
val_accuracy_knn = accuracy_score(val_labels, val_predictions_knn)
print(f"Validation Accuracy (KNN): {val_accuracy_knn * 100:.2f}%")

# Đánh giá mô hình trên bộ testing
test_predictions_knn = knn_classifier.predict(test_features_scaled)
scale_test_accuracy_mfcc_knn = accuracy_score(test_labels, test_predictions_knn)
print(f"Test Accuracy (KNN): {scale_test_accuracy_mfcc_knn * 100:.2f}%")
def pre_emphasis(signal_in, pre_emph=0.97):
    """
    Áp dụng pre-emphasis để nhấn mạnh các tần số cao.
    """
    return np.append(signal_in[0], signal_in[1:] - pre_emph * signal_in[:-1])

def framing(signal_in, sample_rate, frame_size=0.025, frame_stride=0.01):
    """
    Chia tín hiệu thành các frame có kích thước và bước nhảy xác định.
    """
    frame_length = int(round(frame_size * sample_rate))
    frame_step = int(round(frame_stride * sample_rate))
    signal_length = len(signal_in)
    num_frames = int(np.ceil(np.abs(signal_length - frame_length) / frame_step)) + 1

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(signal_in, z)

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    return frames

def windowing(frames):
    """
    Áp dụng cửa sổ Hamming cho mỗi frame.
    """
    frame_length = frames.shape[1]
    hamming = np.hamming(frame_length)
    return frames * hamming

def fft_frames(frames, NFFT=512):
    """
    Tính FFT cho mỗi frame và lấy giá trị magnitude.
    """
    return np.absolute(np.fft.rfft(frames, NFFT))

def compute_fft_features(signal_in, sample_rate, frame_size=0.025, frame_stride=0.01, NFFT=512, apply_log=True):
    """
    Tính toán đặc trưng FFT cho tín hiệu âm thanh:
      - Pre-emphasis, Framing, Windowing.
      - Tính FFT cho từng frame và lấy giá trị magnitude.
      - Trung bình các frame để có vector đặc trưng ổn định.
      - (Tùy chọn) Áp dụng log để giảm phạm vi giá trị.
      
    Trả về: vector đặc trưng có kích thước (NFFT/2+1,).
    """
    emphasized_signal = pre_emphasis(signal_in)
    frames = framing(emphasized_signal, sample_rate, frame_size, frame_stride)
    windowed_frames = windowing(frames)
    mag_frames = fft_frames(windowed_frames, NFFT)
    fft_feature = np.mean(mag_frames, axis=0)  # Trung bình theo các frame
    if apply_log:
        fft_feature = np.log(fft_feature + 1e-8)  # Thêm epsilon để tránh log(0)
    return fft_feature

def load_fft_features_from_directory(directory, sample_rate=22050, NFFT=512):
    """
    Duyệt qua các file âm thanh trong thư mục và tính đặc trưng FFT cho mỗi file.
    Giả sử trong directory có hai thư mục con: 'Queen' và 'NonQueen'.
    
    Trả về:
      - features: mảng đặc trưng (mỗi đặc trưng có kích thước NFFT/2+1)
      - labels: nhãn tương ứng.
    """
    labels = []
    features = []
    for label in ['Queen', 'NonQueen']:
        path = os.path.join(directory, label)
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            signal, sr = librosa.load(file_path, sr=sample_rate)
            fft_feature = compute_fft_features(signal, sr, NFFT=NFFT)
            features.append(fft_feature)
            labels.append(label)
    return np.array(features), np.array(labels)

train_features, train_labels = load_fft_features_from_directory(train_path, sample_rate=22050, NFFT=512)
val_features, val_labels = load_fft_features_from_directory(val_path, sample_rate=22050, NFFT=512)
test_features, test_labels = load_fft_features_from_directory(test_path, sample_rate=22050, NFFT=512)

scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
val_features_scaled = scaler.transform(val_features)
test_features_scaled = scaler.transform(test_features)
scaler = MinMaxScaler()
train_features_mmscaled = scaler.fit_transform(train_features)
val_features_mmscaled = scaler.transform(val_features)
test_features_mmscaled = scaler.transform(test_features)

# KNN without data scaling
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(train_features, train_labels)

val_pred_knn = knn_classifier.predict(val_features)
test_accuracy_fft_knn = knn_classifier.predict(test_features)
print(f"KNN (FFT features) - Validation Accuracy: {accuracy_score(val_labels, val_pred_knn)*100:.2f}%")
print(f"KNN (FFT features) - Test Accuracy: {accuracy_score(test_labels, test_accuracy_fft_knn)*100:.2f}%")
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(train_features_scaled, train_labels)

val_pred_knn = knn_classifier.predict(val_features_scaled)
test_accuracy_fft_knn = knn_classifier.predict(test_features_scaled)
print(f"KNN (FFT features) - Validation Accuracy: {accuracy_score(val_labels, val_pred_knn)*100:.2f}%")
print(f"KNN (FFT features) - Test Accuracy: {accuracy_score(test_labels, test_accuracy_fft_knn)*100:.2f}%")
# Khởi tạo mô hình SVM
svm_rbf_classifier = SVC(C=100, kernel='rbf', degree=3, gamma=0.1, coef0=0.0, shrinking=True, 
                         probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, 
                         max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=42)

# Huấn luyện mô hình SVM
svm_rbf_classifier.fit(train_features, train_labels)

# Đánh giá mô hình trên bộ validation
val_predictions_svm_rbf = svm_rbf_classifier.predict(val_features)
val_accuracy_svm_rbf = accuracy_score(val_labels, val_predictions_svm_rbf)
print(f"Validation Accuracy (SVM with RBF Kernel): {val_accuracy_svm_rbf * 100:.2f}%")

# Đánh giá mô hình trên bộ testing
test_predictions_svm_rbf = svm_rbf_classifier.predict(test_features)
test_accuracy_fft_svm = accuracy_score(test_labels, test_predictions_svm_rbf)
print(f"Test Accuracy (SVM with RBF Kernel): {test_accuracy_fft_svm * 100:.2f}%")
# Khởi tạo mô hình SVM với data scaling
svm_rbf_classifier = SVC(C=100, kernel='rbf', degree=3, gamma=0.1, coef0=0.0, shrinking=True, 
                         probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, 
                         max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=42)

# Huấn luyện mô hình SVM
svm_rbf_classifier.fit(train_features_scaled, train_labels)

# Đánh giá mô hình trên bộ validation
val_predictions_svm_rbf = svm_rbf_classifier.predict(val_features_scaled)
val_accuracy_svm_rbf = accuracy_score(val_labels, val_predictions_svm_rbf)
print(f"Validation Accuracy (SVM with RBF Kernel): {val_accuracy_svm_rbf * 100:.2f}%")

# Đánh giá mô hình trên bộ testing
test_predictions_svm_rbf = svm_rbf_classifier.predict(test_features_scaled)
scale_test_accuracy_fft_svm = accuracy_score(test_labels, test_predictions_svm_rbf)
print(f"Test Accuracy (SVM with RBF Kernel): {scale_test_accuracy_fft_svm * 100:.2f}%")
lr_classifier = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=10, fit_intercept=True, intercept_scaling=1, 
                                   class_weight=None, random_state=42, solver='lbfgs', max_iter=1500, multi_class='deprecated', 
                                   verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
lr_classifier.fit(train_features, train_labels)

val_pred_lr = lr_classifier.predict(val_features)
test_accuracy_fft_lr = lr_classifier.predict(test_features)
print(f"Logistic Regression (FFT features) - Validation Accuracy: {accuracy_score(val_labels, val_pred_lr)*100:.2f}%")
print(f"Logistic Regression (FFT features) - Test Accuracy: {accuracy_score(test_labels, test_accuracy_fft_lr)*100:.2f}%")

lr_classifier = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=10, fit_intercept=True, intercept_scaling=1, 
                                   class_weight=None, random_state=42, solver='saga', max_iter=1000, multi_class='deprecated', 
                                   verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)

lr_classifier.fit(train_features_mmscaled, train_labels)

val_pred_lr = lr_classifier.predict(val_features_mmscaled)
scale_test_accuracy_fft_lr = lr_classifier.predict(test_features_mmscaled)
print(f"Logistic Regression (FFT features) - Validation Accuracy: {accuracy_score(val_labels, val_pred_lr)*100:.2f}%")
print(f"Logistic Regression (FFT features) - Test Accuracy: {accuracy_score(test_labels, scale_test_accuracy_fft_lr)*100:.2f}%")

rf_classifier = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, 
                                       min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', 
                                       max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, 
                                       n_jobs=None, random_state=42, verbose=0, warm_start=False, class_weight=None, 
                                       ccp_alpha=0.0, max_samples=None, monotonic_cst=None)

rf_classifier.fit(train_features, train_labels)

val_pred_rf = rf_classifier.predict(val_features)
test_accuracy_fft_rf = rf_classifier.predict(test_features)
print(f"Random Forest (FFT features) - Validation Accuracy: {accuracy_score(val_labels, val_pred_rf)*100:.2f}%")
print(f"Random Forest (FFT features) - Test Accuracy: {accuracy_score(test_labels, test_accuracy_fft_rf)*100:.2f}%")
# Khởi tạo mô hình Extra Trees 
et_classifier = ExtraTreesClassifier(n_estimators=200, criterion='gini', max_depth=30, min_samples_split=2, min_samples_leaf=1, 
                                     min_weight_fraction_leaf=0.0, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0,
                                     bootstrap=False, oob_score=False, n_jobs=None, random_state=42, verbose=0, warm_start=False, 
                                     class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)

# Huấn luyện mô hình Extra Trees với dữ liệu đã chuẩn hóa
et_classifier.fit(train_features, train_labels)

# Đánh giá mô hình trên bộ validation
val_predictions_et = et_classifier.predict(val_features)
val_accuracy_et = accuracy_score(val_labels, val_predictions_et)
print(f"Validation Accuracy (Extra Trees): {val_accuracy_et * 100:.2f}%")

# Đánh giá mô hình trên bộ testing
test_predictions_et = et_classifier.predict(test_features)
test_accuracy_fft_et = accuracy_score(test_labels, test_predictions_et)
print(f"Test Accuracy (Extra Trees): {test_accuracy_fft_et * 100:.2f}%")

# Khởi tạo mô hình Extra Trees với data scaling
et_classifier = ExtraTreesClassifier(n_estimators=200, criterion='gini', max_depth=30, min_samples_split=2, min_samples_leaf=1, 
                                     min_weight_fraction_leaf=0.0, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0,
                                     bootstrap=False, oob_score=False, n_jobs=None, random_state=42, verbose=0, warm_start=False, 
                                     class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)

# Huấn luyện mô hình Extra Trees với dữ liệu đã chuẩn hóa
et_classifier.fit(train_features_scaled, train_labels)

# Đánh giá mô hình trên bộ validation
val_predictions_et = et_classifier.predict(val_features_scaled)
val_accuracy_et = accuracy_score(val_labels, val_predictions_et)
print(f"Validation Accuracy (Extra Trees): {val_accuracy_et * 100:.2f}%")

# Đánh giá mô hình trên bộ testing
test_predictions_et = et_classifier.predict(test_features_scaled)
scale_test_accuracy_fft_et = accuracy_score(test_labels, test_predictions_et)
print(f"Test Accuracy (Extra Trees): {scale_test_accuracy_fft_et * 100:.2f}%")

import numpy as np
import librosa
import os

def apply_preprocessing(signal_in, sample_rate):
    """
    Apply optional preprocessing steps on the signal:
      - Apply pre-emphasis filtering.
    """
    # Pre-emphasis filter: Boosting higher frequencies slightly
    pre_emphasis = 0.97
    signal_in = np.append(signal_in[0], signal_in[1:] - pre_emphasis * signal_in[:-1])
    return signal_in

def compute_stft_features(signal_in, sample_rate, n_fft=2048, hop_length=512, apply_log=True):
    """
    Compute STFT features with improvements:
      - Apply windowing (Hanning window).
      - Optional log scaling with dynamic range compression.
      - Optional normalization and energy-based features.
    """
    # Apply preprocessing (e.g., pre-emphasis)
    signal_in = apply_preprocessing(signal_in, sample_rate)
    
    # Compute STFT with a window function to avoid spectral leakage
    stft_matrix = np.abs(librosa.stft(signal_in, n_fft=n_fft, hop_length=hop_length, window='hann'))
    
    # Optional: Apply logarithmic scaling (dynamic range compression)
    if apply_log:
        stft_matrix = np.log(stft_matrix + 1e-10)  # More robust epsilon for log scaling

    # Feature extraction: Mean and additional energy-based features (e.g., variance)
    features_mean = np.mean(stft_matrix, axis=1)
    features_variance = np.var(stft_matrix, axis=1)
    
    # Combine mean and variance (you could also add other stats like median or skewness)
    features = np.concatenate([features_mean, features_variance], axis=0)

    # Normalize the features (optional)
    features = (features - np.mean(features)) / np.std(features)  # Z-score normalization

    return features

def load_stft_features_from_directory(directory, sample_rate=22050, n_fft=2048, hop_length=512):
    """
    Load and compute STFT features from audio files in a directory, with improved accuracy.
    Assumes 'Queen' and 'NonQueen' subdirectories are present.
    """
    features = []
    labels = []

    for label in ['Queen', 'NonQueen']:
        path = os.path.join(directory, label)
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            # Load audio file
            signal, sr = librosa.load(file_path, sr=sample_rate)
            
            # Compute STFT features
            stft_feature = compute_stft_features(signal, sr, n_fft=n_fft, hop_length=hop_length, apply_log=True)
            
            features.append(stft_feature)
            labels.append(label)

    return np.array(features), np.array(labels)


train_features_stft, train_labels_stft = load_stft_features_from_directory(train_path, sample_rate=22050, n_fft=2048, hop_length=512)
val_features_stft, val_labels_stft     = load_stft_features_from_directory(val_path, sample_rate=22050, n_fft=2048, hop_length=512)
test_features_stft, test_labels_stft   = load_stft_features_from_directory(test_path, sample_rate=22050, n_fft=2048, hop_length=512)

scaler_stft = StandardScaler()
train_features_stft_scaled = scaler_stft.fit_transform(train_features_stft)
val_features_stft_scaled   = scaler_stft.transform(val_features_stft)
test_features_stft_scaled  = scaler_stft.transform(test_features_stft)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

knn_classifier_stft = KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto', leaf_size=30, p=2, 
                                           metric='minkowski', metric_params=None, n_jobs=None)

knn_classifier_stft.fit(train_features_stft, train_labels_stft)
val_pred_stft_knn = knn_classifier_stft.predict(val_features_stft)
test_pred_stft_knn = knn_classifier_stft.predict(test_features_stft)

test_accuracy_stft_knn = accuracy_score(test_labels_stft, test_pred_stft_knn)
print(f"KNN (STFT) - Test Accuracy: {test_accuracy_stft_knn*100:.2f}%")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

knn_classifier_stft = KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto', leaf_size=30, p=2, 
                                           metric='minkowski', metric_params=None, n_jobs=None)

knn_classifier_stft.fit(train_features_stft_scaled, train_labels_stft)
val_pred_stft_knn = knn_classifier_stft.predict(val_features_stft_scaled)
test_pred_stft_knn = knn_classifier_stft.predict(test_features_stft_scaled)

test_accuracy_stft_knn = accuracy_score(test_labels_stft, test_pred_stft_knn)
print(f"KNN (STFT) - Test Accuracy: {test_accuracy_stft_knn*100:.2f}%")

# Khởi tạo mô hình SVM với data scaling
svm_rbf_classifier = SVC(C=10, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, 
                         probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, 
                         max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=42)

# Huấn luyện mô hình SVM
svm_rbf_classifier.fit(train_features_stft_scaled, train_labels_stft)

# Đánh giá mô hình trên bộ validation
val_predictions_svm_rbf = svm_rbf_classifier.predict(val_features_stft_scaled)
val_accuracy_stft_svm = accuracy_score(val_labels_stft, val_predictions_svm_rbf)
print(f"Validation Accuracy (SVM with RBF Kernel): {val_accuracy_stft_svm * 100:.2f}%")

# Đánh giá mô hình trên bộ testing
test_predictions_svm_rbf = svm_rbf_classifier.predict(test_features_stft_scaled)
scale_test_accuracy_stft_svm = accuracy_score(test_labels_stft, test_predictions_svm_rbf)
print(f"Test Accuracy (SVM with RBF Kernel): {scale_test_accuracy_stft_svm * 100:.2f}%")
from sklearn.linear_model import LogisticRegression

lr_classifier_stft = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear', penalty= 'l2', C= 0.08858667904100823)
lr_classifier_stft.fit(train_features_stft_scaled, train_labels_stft)
val_pred_stft_lr = lr_classifier_stft.predict(val_features_stft_scaled)
test_pred_stft_lr = lr_classifier_stft.predict(test_features_stft_scaled)

test_accuracy_stft_lr = accuracy_score(test_labels_stft, test_pred_stft_lr)
print(f"Logistic Regression (STFT) - Test Accuracy: {test_accuracy_stft_lr*100:.2f}%")
rf_classifier_stft = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, 
                                       min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
                                       max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, 
                                       bootstrap=False, oob_score=False, n_jobs=None, random_state=42, verbose=0, 
                                       warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)

rf_classifier_stft.fit(train_features_stft, train_labels_stft)
val_pred_stft_rf = rf_classifier_stft.predict(val_features_stft)
test_pred_stft_rf = rf_classifier_stft.predict(test_features_stft)

test_accuracy_stft_rf = accuracy_score(test_labels_stft, test_pred_stft_rf)
print(f"Random Forest (STFT) - Test Accuracy: {test_accuracy_stft_rf*100:.2f}%")

et_classifier_stft = ExtraTreesClassifier(n_estimators=100, random_state=42)
et_classifier_stft.fit(train_features_stft_scaled, train_labels_stft)
val_pred_stft_et = et_classifier_stft.predict(val_features_stft_scaled)
test_pred_stft_et = et_classifier_stft.predict(test_features_stft_scaled)

test_accuracy_stft_et = accuracy_score(test_labels_stft, test_pred_stft_et)
print(f"Extra Trees (STFT) - Test Accuracy: {test_accuracy_stft_et*100:.2f}%")

def compute_cqt_features(signal_in, sample_rate, hop_length=512, fmin=None, 
                         n_bins=84, bins_per_octave=12, apply_log=True):
    """
    Tính đặc trưng CQT của tín hiệu:
      - Sử dụng librosa.cqt để tính ma trận CQT, lấy giá trị magnitude.
      - (Tùy chọn) Áp dụng log scaling để giảm phạm vi giá trị.
      - Trung bình theo trục thời gian (axis=1) để thu được vector đặc trưng có kích thước (n_bins,).
      
    Nếu fmin không được chỉ định, librosa sẽ tự động sử dụng fmin mặc định.
    """
    cqt_matrix = np.abs(librosa.cqt(signal_in, sr=sample_rate, hop_length=hop_length,
                                    fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave))
    if apply_log:
        cqt_matrix = np.log(cqt_matrix + 1e-8)
    # Trung bình theo trục thời gian (tính trung bình các cột)
    features = np.mean(cqt_matrix, axis=1)
    return features

def load_cqt_features_from_directory(directory, sample_rate=22050, 
                                     hop_length=512, fmin=None, n_bins=84, bins_per_octave=12):
    """
    Duyệt qua các file âm thanh trong thư mục và tính đặc trưng CQT cho mỗi file.
    Giả sử trong directory có 2 thư mục con: 'Queen' và 'NonQueen'.
    
    Trả về:
      - features: mảng các vector đặc trưng (mỗi vector có kích thước n_bins)
      - labels: nhãn tương ứng.
    """
    import os
    features = []
    labels = []
    for label in ['Queen', 'NonQueen']:
        path = os.path.join(directory, label)
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            signal, sr = librosa.load(file_path, sr=sample_rate)
            cqt_feature = compute_cqt_features(signal, sr, hop_length=hop_length, 
                                               fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave, 
                                               apply_log=True)
            features.append(cqt_feature)
            labels.append(label)
    return np.array(features), np.array(labels)

train_features_cqt, train_labels_cqt = load_cqt_features_from_directory(train_path, sample_rate=22050,
                                                                        hop_length=512, fmin=None, n_bins=84, bins_per_octave=12)
val_features_cqt, val_labels_cqt     = load_cqt_features_from_directory(val_path, sample_rate=22050,
                                                                        hop_length=512, fmin=None, n_bins=84, bins_per_octave=12)
test_features_cqt, test_labels_cqt   = load_cqt_features_from_directory(test_path, sample_rate=22050,
                                                                        hop_length=512, fmin=None, n_bins=84, bins_per_octave=12)

scaler_cqt = StandardScaler()
train_features_cqt_scaled = scaler_cqt.fit_transform(train_features_cqt)
val_features_cqt_scaled   = scaler_cqt.transform(val_features_cqt)
test_features_cqt_scaled  = scaler_cqt.transform(test_features_cqt)
knn_classifier_cqt = KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='auto', leaf_size=30, p=2, 
                                      metric='minkowski', metric_params=None, n_jobs=None)

knn_classifier_cqt.fit(train_features_cqt_scaled, train_labels_cqt)
val_pred_cqt_knn = knn_classifier_cqt.predict(val_features_cqt_scaled)
test_pred_cqt_knn = knn_classifier_cqt.predict(test_features_cqt_scaled)

test_accuracy_cqt_knn = accuracy_score(test_labels_cqt, test_pred_cqt_knn)
print(f"KNN (CQT) - Test Accuracy: {test_accuracy_cqt_knn*100:.2f}%")

svm_classifier_cqt = SVC(C=100, kernel='rbf', gamma='auto', random_state=42)
svm_classifier_cqt.fit(train_features_cqt_scaled, train_labels_cqt)
val_pred_cqt_svm = svm_classifier_cqt.predict(val_features_cqt_scaled)
test_pred_cqt_svm = svm_classifier_cqt.predict(test_features_cqt_scaled)

test_accuracy_cqt_svm = accuracy_score(test_labels_cqt, test_pred_cqt_svm)
print(f"SVM (CQT) - Test Accuracy: {test_accuracy_cqt_svm*100:.2f}%")

lr_classifier_cqt =LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, 
                                      fit_intercept=True, intercept_scaling=1, 
                                      class_weight=None, random_state=None, solver='lbfgs', 
                                      max_iter=500, multi_class='deprecated', verbose=0, 
                                      warm_start=False, n_jobs=None, l1_ratio=None)

lr_classifier_cqt.fit(train_features_cqt_scaled, train_labels_cqt)
val_pred_cqt_lr = lr_classifier_cqt.predict(val_features_cqt_scaled)
test_pred_cqt_lr = lr_classifier_cqt.predict(test_features_cqt_scaled)

test_accuracy_cqt_lr = accuracy_score(test_labels_cqt, test_pred_cqt_lr)
print(f"Logistic Regression (CQT) - Test Accuracy: {test_accuracy_cqt_lr*100:.2f}%")

rf_classifier_cqt = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, 
                                       min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
                                       max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, 
                                       bootstrap=False, oob_score=False, n_jobs=None, random_state=42, verbose=0, 
                                       warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)

rf_classifier_cqt.fit(train_features_cqt, train_labels_cqt)
val_pred_cqt_rf = rf_classifier_cqt.predict(val_features_cqt)
test_pred_cqt_rf = rf_classifier_cqt.predict(test_features_cqt)

test_accuracy_cqt_rf = accuracy_score(test_labels_cqt, test_pred_cqt_rf)
print(f"Random Forest (CQT) - Test Accuracy: {test_accuracy_cqt_rf*100:.2f}%")

et_classifier_cqt = ExtraTreesClassifier(n_estimators=200, criterion='gini', max_depth=30, min_samples_split=2, min_samples_leaf=1, 
                                     min_weight_fraction_leaf=0.0, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0,
                                     bootstrap=False, oob_score=False, n_jobs=None, random_state=42, verbose=0, warm_start=False, 
                                     class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)

et_classifier_cqt.fit(train_features_cqt_scaled, train_labels_cqt)
val_pred_cqt_et = et_classifier_cqt.predict(val_features_cqt_scaled)
test_pred_cqt_et = et_classifier_cqt.predict(test_features_cqt_scaled)

test_accuracy_cqt_et = accuracy_score(test_labels_cqt, test_pred_cqt_et)
print(f"Extra Trees (CQT) - Test Accuracy: {test_accuracy_cqt_et*100:.2f}%")
import numpy as np
import librosa

def chroma_features(x, sr, n_fft=2048, hop_length=512, n_chroma=12):
    """
    Tính toán đặc trưng Chroma của tín hiệu x.
    
    Args:
    - x: Tín hiệu âm thanh đầu vào (mảng numpy)
    - sr: Tỷ lệ lấy mẫu (sampling rate)
    - n_fft: Kích thước cửa sổ FFT
    - hop_length: Bước nhảy (hop size) cho FFT
    - n_chroma: Số lượng lớp âm (pitch classes), thường là 12 (C, C#, D, D#, E, F, F#, G, G#, A, A#, B)
    
    Returns:
    - chroma: Mảng Chroma của tín hiệu âm thanh
    """
    # Tính toán STFT (Short-Time Fourier Transform)
    D = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))**2

    # Tính toán đặc trưng Chroma
    chroma = librosa.feature.chroma_stft(S=D, sr=sr, n_chroma=n_chroma)

    return chroma

def load_chroma_features_from_directory(directory, sample_rate=22050, n_fft=2048, hop_length=512, n_chroma=12):
    """
    Duyệt qua các file âm thanh trong thư mục và tính đặc trưng Chroma cho mỗi file.
    Giả sử trong directory có 2 thư mục con: 'Queen' và 'NonQueen'.
    
    Trả về:
      - features: mảng các vector đặc trưng (mỗi vector có kích thước n_chroma)
      - labels: nhãn tương ứng.
    """
    import os
    features = []
    labels = []
    for label in ['Queen', 'NonQueen']:
        path = os.path.join(directory, label)
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            signal, sr = librosa.load(file_path, sr=sample_rate)
            chroma_feature = chroma_features(signal, sr, n_fft=n_fft, 
                                                     hop_length=hop_length, n_chroma=n_chroma)
            features.append(chroma_feature)
            labels.append(label)
    return np.array(features), np.array(labels)

train_features_chroma, train_labels_chroma = load_chroma_features_from_directory(train_path, sample_rate=None, n_fft=2048, hop_length=512, n_chroma=12)
val_features_chroma, val_labels_chroma     = load_chroma_features_from_directory(val_path, sample_rate=None, n_fft=2048, hop_length=512, n_chroma=12)
test_features_chroma, test_labels_chroma   = load_chroma_features_from_directory(test_path, sample_rate=None, n_fft=2048, hop_length=512, n_chroma=12)
# Reshape the features to 2D before scaling (flatten the chroma features)
train_features_chroma_reshaped = train_features_chroma.reshape(train_features_chroma.shape[0], -1)
val_features_chroma_reshaped = val_features_chroma.reshape(val_features_chroma.shape[0], -1)
test_features_chroma_reshaped = test_features_chroma.reshape(test_features_chroma.shape[0], -1)

# Apply scaling
scaler_chroma = StandardScaler()
train_features_chroma_scaled = scaler_chroma.fit_transform(train_features_chroma_reshaped)
val_features_chroma_scaled   = scaler_chroma.transform(val_features_chroma_reshaped)
test_features_chroma_scaled  = scaler_chroma.transform(test_features_chroma_reshaped)

knn_classifier_chroma = KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto', leaf_size=30, p=2, 
                                      metric='minkowski', metric_params=None, n_jobs=None)

knn_classifier_chroma.fit(train_features_chroma_scaled, train_labels_chroma)
val_pred_chroma_knn = knn_classifier_chroma.predict(val_features_chroma_scaled)
test_pred_chroma_knn = knn_classifier_chroma.predict(test_features_chroma_scaled)

test_accuracy_chroma_knn = accuracy_score(test_labels_chroma, test_pred_chroma_knn)
print(f"KNN (Chroma) - Test Accuracy: {test_accuracy_chroma_knn*100:.2f}%")
