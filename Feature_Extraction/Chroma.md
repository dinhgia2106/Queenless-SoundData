Giải thích chi tiết từng bước và từng dòng mã trong chương trình tính toán **Chroma** và cách dữ liệu đầu vào được xử lý qua các thuật toán, cũng như đầu ra của mỗi bước.

### 1. **Định nghĩa hàm `chroma_features`**

```python
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
```

- **`chroma_features`** là hàm tính toán đặc trưng **Chroma** của tín hiệu âm thanh đầu vào `x`.
- Các tham số:
  - **`x`**: Tín hiệu âm thanh dưới dạng mảng `numpy`, thường được đọc từ các file âm thanh.
  - **`sr`**: Tỷ lệ lấy mẫu (sampling rate) của tín hiệu âm thanh. Tỷ lệ này cần được giữ nguyên khi tải tín hiệu âm thanh.
  - **`n_fft`**: Kích thước của cửa sổ FFT (số mẫu trong mỗi khung). FFT sẽ chia tín hiệu thành các phần nhỏ để tính toán các thành phần tần số.
  - **`hop_length`**: Bước nhảy giữa các khung trong quá trình tính toán FFT. Đây là số mẫu mà cửa sổ FFT di chuyển sau mỗi lần tính toán.
  - **`n_chroma`**: Số lượng lớp âm (pitch classes), thường là 12 (tương ứng với các nốt nhạc trong một quãng tám từ C đến B).

### 2. **Tính toán STFT (Short-Time Fourier Transform)**

```python
    D = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))**2
```

- **`librosa.stft(x, n_fft=n_fft, hop_length=hop_length)`**: Tính toán **Short-Time Fourier Transform (STFT)** của tín hiệu âm thanh `x`. STFT chuyển tín hiệu từ miền thời gian sang miền tần số, giúp phân tích tín hiệu theo các khung thời gian nhỏ.

  - **`n_fft`**: Kích thước cửa sổ FFT. Đây là số mẫu mà mỗi cửa sổ FFT sẽ xử lý.
  - **`hop_length`**: Bước nhảy giữa các cửa sổ FFT, quyết định độ phân giải thời gian.

- **`np.abs(... )**2`**: Lấy giá trị tuyệt đối của các hệ số Fourier và bình phương chúng để tính toán **năng lượng phổ tần số\*\* (spectral energy). Năng lượng này sẽ giúp xác định độ mạnh của các tần số trong tín hiệu.

### 3. **Tính toán đặc trưng Chroma**

```python
    chroma = librosa.feature.chroma_stft(S=D, sr=sr, n_chroma=n_chroma)
```

- **`librosa.feature.chroma_stft`**: Hàm này tính toán đặc trưng **Chroma** từ phổ tần số `D`.
  - **`S=D`**: Sử dụng năng lượng phổ tần số `D` đã được tính toán từ STFT.
  - **`sr=sr`**: Tỷ lệ lấy mẫu của tín hiệu âm thanh.
  - **`n_chroma=n_chroma`**: Số lớp âm (pitch classes) mà chúng ta muốn tính, thường là 12, tương ứng với các nốt nhạc trong một quãng tám.

Hàm này sử dụng **STFT** và ánh xạ các thành phần tần số vào các lớp âm **Chroma**. Các giá trị trong `chroma` tương ứng với các nốt nhạc (C, C#, D, D#, E, F, F#, G, G#, A, A#, B).

### 4. **Trả về đặc trưng Chroma**

```python
    return chroma
```

- Hàm trả về mảng **`chroma`** chứa đặc trưng Chroma của tín hiệu âm thanh, với mỗi cột đại diện cho một khung thời gian và mỗi hàng đại diện cho một lớp âm.

### 5. **Đọc tín hiệu âm thanh và xử lý**

```python
audio_path = 'your_audio_file.wav'
signal, rate = librosa.load(audio_path, sr=None)  # sr=None để giữ nguyên tần số lấy mẫu
```

- **`librosa.load(audio_path, sr=None)`**: Đọc tín hiệu âm thanh từ file `.wav`.
  - **`sr=None`**: Giữ nguyên tỷ lệ lấy mẫu (sampling rate) của tín hiệu âm thanh gốc thay vì chuyển đổi sang một giá trị mặc định.

### 6. **Tính toán đặc trưng Chroma**

```python
chroma = chroma_features(signal, rate)
```

- Gọi hàm **`chroma_features`** để tính toán đặc trưng **Chroma** của tín hiệu âm thanh `signal` với tỷ lệ lấy mẫu `rate`.

### 7. **Vẽ biểu đồ Chroma**

```python
plt.figure(figsize=(10, 6))
librosa.display.specshow(chroma, x_axis='time', y_axis='chroma', cmap='coolwarm')
plt.colorbar(label='Chroma Magnitude')
plt.title('Chroma Features of Audio Signal')
plt.xlabel('Time (s)')
plt.ylabel('Pitch Class')
plt.tight_layout()
plt.show()
```

- **`plt.figure(figsize=(10, 6))`**: Tạo một cửa sổ đồ họa với kích thước 10x6 inch.
- **`librosa.display.specshow(chroma, x_axis='time', y_axis='chroma', cmap='coolwarm')`**: Vẽ biểu đồ đặc trưng **Chroma** sử dụng **`specshow`** của `librosa`. Biểu đồ này sẽ hiển thị sự thay đổi của các lớp âm theo thời gian.
  - **`x_axis='time'`**: Trục hoành là thời gian.
  - **`y_axis='chroma'`**: Trục tung là các lớp âm (pitch classes).
  - **`cmap='coolwarm'`**: Áp dụng bảng màu đẹp cho biểu đồ.
- **`plt.colorbar(label='Chroma Magnitude')`**: Thêm thanh màu vào biểu đồ để hiển thị độ lớn của các giá trị **Chroma**.
- **`plt.show()`**: Hiển thị biểu đồ.

### Đầu ra của từng bước:

1. **`chroma`**: Mảng đặc trưng **Chroma** của tín hiệu âm thanh, chứa các giá trị độ mạnh của các nốt nhạc (C, C#, D, ...) tại mỗi thời điểm.
2. **Biểu đồ Chroma**: Biểu đồ này hiển thị sự thay đổi của các nốt nhạc (pitch classes) theo thời gian, giúp phân tích cấu trúc âm nhạc trong tín hiệu âm thanh.

### Tóm tắt:

- **Chroma** là một phương pháp để phân tích các đặc trưng âm sắc và hài hòa của tín hiệu âm thanh, giúp xác định các nốt nhạc và cao độ trong tín hiệu.
- Các bước chính bao gồm tính toán **STFT**, áp dụng **Mel filter bank** để ánh xạ các tần số vào các nốt nhạc, và tính toán **Chroma** từ phổ tần số.
- Biểu đồ Chroma giúp ta phân tích đặc trưng âm sắc của tín hiệu âm thanh theo các nốt nhạc, điều này rất hữu ích trong phân tích nhạc và nhận dạng âm thanh.
