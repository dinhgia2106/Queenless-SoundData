Giải thích chi tiết từng bước và từng dòng mã trong chương trình tính toán **Spectral Contrast (SC)**, cách dữ liệu đầu vào được xử lý qua các thuật toán và đầu ra của mỗi bước.

### 1. **Định nghĩa hàm `spectral_contrast`**:

```python
def spectral_contrast(x, sr, n_bands=6, fmin=200, fmax=8000, n_fft=2048, hop_length=512):
    """
    Tính toán Spectral Contrast (SC) của tín hiệu x.

    Args:
    - x: Tín hiệu âm thanh đầu vào (mảng numpy)
    - sr: Tỷ lệ lấy mẫu (sampling rate)
    - n_bands: Số lượng dải tần số (frequency bands)
    - fmin: Tần số bắt đầu của dải tần số thấp
    - fmax: Tần số kết thúc của dải tần số cao
    - n_fft: Kích thước cửa sổ FFT
    - hop_length: Bước nhảy (hop size) cho FFT

    Returns:
    - sc: Spectral Contrast (SC) của tín hiệu
    """
```

- **`spectral_contrast`** là hàm chính thực hiện **Spectral Contrast (SC)** của tín hiệu đầu vào `x`.
- Các tham số:
  - **`x`**: Tín hiệu âm thanh đầu vào dưới dạng mảng `numpy`.
  - **`sr`**: Tỷ lệ lấy mẫu (sampling rate) của tín hiệu âm thanh.
  - **`n_bands`**: Số lượng dải tần số (frequency bands) để tính toán SC.
  - **`fmin` và `fmax`**: Các giá trị tần số bắt đầu và kết thúc của các dải tần số.
  - **`n_fft`**: Kích thước cửa sổ FFT (số mẫu trong mỗi khung).
  - **`hop_length`**: Bước nhảy giữa các khung tín hiệu khi tính FFT.

### 2. **Tính toán Mel filter bank**:

```python
    mel_filter = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_bands, fmin=fmin, fmax=fmax)
```

- **`librosa.filters.mel`**: Hàm này tạo ra một **Mel filter bank** để chia tín hiệu thành các dải tần số.

  - **`sr`**: Tỷ lệ lấy mẫu của tín hiệu âm thanh.
  - **`n_fft`**: Kích thước của cửa sổ FFT.
  - **`n_mels`**: Số lượng dải tần số Mel (tương đương với số lượng bin tần số trong SC).
  - **`fmin` và `fmax`**: Các giá trị tần số bắt đầu và kết thúc cho các dải tần số.

- **Mel filter bank** là bộ lọc phân chia phổ tần số thành các dải tần số tương tự như cách mà tai người nghe phân biệt âm thanh. Mỗi dải tần số trong Mel filter bank sẽ có các bộ lọc tương ứng với các tần số Mel.

### 3. **Tính toán phổ tần số**:

```python
    D = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))**2
```

- **`librosa.stft`**: Hàm này tính toán **Short-Time Fourier Transform (STFT)** của tín hiệu âm thanh.

  - **`n_fft`**: Kích thước cửa sổ FFT (số mẫu trong mỗi khung).
  - **`hop_length`**: Bước nhảy giữa các khung (tức là độ dài bước di chuyển cửa sổ FFT).

- **`np.abs(... )`**: Lấy giá trị tuyệt đối của các hệ số Fourier.
- **`**2`\*\*: Bình phương các giá trị Fourier để tính năng lượng phổ.

Kết quả **`D`** là phổ tần số của tín hiệu âm thanh, thể hiện độ lớn (magnitude) của tín hiệu trong từng khung.

### 4. **Áp dụng Mel filter bank vào phổ tần số**:

```python
    mel_spectrum = np.dot(mel_filter, D)
```

- **`np.dot(mel_filter, D)`**: Áp dụng **Mel filter bank** vào phổ tần số `D` để tính toán năng lượng của tín hiệu trong các dải tần số Mel.
  - **`mel_spectrum`** là mảng chứa năng lượng trong các dải tần số Mel.

### 5. **Tính toán Spectral Contrast (SC)**:

```python
    sc = []
    for i in range(n_bands):
        peak = np.max(mel_spectrum[i, :])
        valley = np.min(mel_spectrum[i, :])
        sc.append(peak - valley)
```

- **`sc`**: Mảng lưu trữ giá trị Spectral Contrast cho mỗi dải tần số.
- **`peak`**: Tính giá trị đỉnh (maximum) trong dải tần số **i**.
- **`valley`**: Tính giá trị đáy (minimum) trong dải tần số **i**.
- **`sc.append(peak - valley)`**: Tính sự khác biệt giữa đỉnh và đáy của dải tần số, là Spectral Contrast (SC) của dải đó.

### 6. **Trả về kết quả SC**:

```python
    return np.array(sc)
```

- Hàm trả về mảng `sc`, chứa Spectral Contrast (SC) của tín hiệu âm thanh.

### 7. **Đọc tín hiệu âm thanh và xử lý**:

```python
audio_path = 'your_audio_file.wav'
signal, rate = librosa.load(audio_path, sr=None)  # sr=None để giữ nguyên tần số lấy mẫu
```

- **`librosa.load`**: Hàm này đọc tín hiệu âm thanh từ file `.wav` và trả về tín hiệu âm thanh dưới dạng mảng `numpy`.
  - **`sr=None`**: Để giữ nguyên tỷ lệ lấy mẫu gốc của tín hiệu âm thanh.

### 8. **Tính toán Spectral Contrast**:

```python
sc = spectral_contrast(signal, rate)
```

- **`spectral_contrast(signal, rate)`**: Gọi hàm `spectral_contrast` để tính toán Spectral Contrast (SC) của tín hiệu âm thanh `signal`.

### 9. **Vẽ biểu đồ Spectral Contrast**:

```python
plt.figure(figsize=(10, 6))
plt.plot(sc)
plt.title('Spectral Contrast (SC) of Audio Signal')
plt.xlabel('Frequency Bands')
plt.ylabel('Spectral Contrast (SC)')
plt.tight_layout()
plt.show()
```

- **`plt.figure(figsize=(10, 6))`**: Tạo một cửa sổ đồ họa với kích thước 10x6 inch.
- **`plt.plot(sc)`**: Vẽ biểu đồ Spectral Contrast (SC) với trục hoành là các dải tần số và trục tung là giá trị SC.
- **`plt.title`** và **`plt.xlabel`, `plt.ylabel`**: Thêm tiêu đề và nhãn cho các trục.
- **`plt.tight_layout()`**: Căn chỉnh lại các yếu tố trong đồ họa để không bị chồng chéo.
- **`plt.show()`**: Hiển thị đồ thị.

### Đầu ra của từng bước:

- **`sc`**: Mảng chứa giá trị **Spectral Contrast (SC)** cho mỗi dải tần số.
- **Biểu đồ**: Biểu đồ hiển thị Spectral Contrast của tín hiệu âm thanh, giúp phân tích sự tương phản giữa các dải tần số.

### Tóm tắt:

Chương trình này tính toán **Spectral Contrast (SC)** của tín hiệu âm thanh để phân tích sự khác biệt giữa các đỉnh và đáy trong phổ tần số. SC được tính bằng cách chia phổ tần số thành các dải tần số sử dụng **Mel filter bank**, sau đó tính toán năng lượng trong các dải tần số và so sánh năng lượng giữa các dải tần số liền kề để tính toán SC. Biểu đồ SC giúp chúng ta nhận diện sự khác biệt giữa các đặc trưng âm thanh trong các dải tần số khác nhau.
