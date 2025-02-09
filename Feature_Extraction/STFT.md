Giải thích chi tiết từng bước và từng dòng mã trong chương trình tính toán **Short-Time Fourier Transform (STFT)**, từ việc xử lý dữ liệu đầu vào cho đến đầu ra của từng bước:

### 1. **Các thư viện cần thiết**:

```python
import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.signal import stft, get_window
```

- **`numpy`**: Thư viện hỗ trợ các phép toán số học với mảng (array) và ma trận. Được sử dụng để thao tác với dữ liệu tín hiệu.
- **`matplotlib.pyplot`**: Thư viện vẽ biểu đồ và đồ họa. Chúng ta sử dụng nó để vẽ **spectrogram** từ kết quả STFT.
- **`librosa`**: Thư viện xử lý tín hiệu âm thanh, được dùng để tải và xử lý các tệp âm thanh.
- **`scipy.signal.stft`**: Hàm từ thư viện `scipy` dùng để tính toán STFT của tín hiệu.
- **`scipy.signal.get_window`**: Hàm này giúp tạo các cửa sổ (window) với các loại khác nhau như `hann`, `hamming`, v.v.

### 2. **Hàm `compute_stft`**:

```python
def compute_stft(signal_in, sample_rate, window_size=2048, hop_size=512, window_type='hann'):
    """
    Hàm tính Short-Time Fourier Transform (STFT) của tín hiệu âm thanh.

    Args:
    - signal_in: Tín hiệu âm thanh đầu vào (mảng numpy)
    - sample_rate: Tỷ lệ lấy mẫu của tín hiệu (Hz)
    - window_size: Kích thước cửa sổ (số mẫu)
    - hop_size: Bước nhảy giữa các khung (số mẫu)
    - window_type: Loại cửa sổ sử dụng ('hann', 'hamming', v.v.)

    Returns:
    - Zxx: Kết quả STFT (biến đổi Fourier trong miền thời gian-tần số)
    """
```

- Hàm `compute_stft` nhận vào các tham số:
  - **`signal_in`**: Mảng numpy chứa tín hiệu âm thanh đầu vào.
  - **`sample_rate`**: Tỷ lệ lấy mẫu của tín hiệu âm thanh (Hz).
  - **`window_size`**: Kích thước cửa sổ (số mẫu) được sử dụng trong mỗi đoạn tín hiệu.
  - **`hop_size`**: Bước nhảy (số mẫu) giữa các khung cửa sổ.
  - **`window_type`**: Loại cửa sổ sử dụng để làm giảm hiệu ứng biên khi thực hiện FFT (ví dụ: 'hann', 'hamming').

#### Tiến hành xử lý tín hiệu:

```python
    # Áp dụng cửa sổ vào tín hiệu
    window = get_window(window_type, window_size)
```

- **`get_window(window_type, window_size)`**: Tạo một cửa sổ (window) với kiểu `window_type` và kích thước `window_size`. Các cửa sổ như **Hanning** hoặc **Hamming** giúp giảm nhiễu biên và cải thiện độ chính xác của FFT khi tín hiệu được chia thành các đoạn nhỏ.

```python
    # Tính toán STFT bằng hàm scipy.signal.stft
    f, t, Zxx = stft(signal_in, fs=sample_rate, window=window, nperseg=window_size, noverlap=hop_size)
```

- **`stft()`**: Tính toán STFT của tín hiệu âm thanh.
  - **`signal_in`**: Tín hiệu âm thanh đầu vào.
  - **`fs=sample_rate`**: Tỷ lệ lấy mẫu (được lấy từ dữ liệu đầu vào).
  - **`window=window`**: Cửa sổ được áp dụng cho tín hiệu.
  - **`nperseg=window_size`**: Số mẫu trong mỗi cửa sổ.
  - **`noverlap=hop_size`**: Số mẫu chồng lắp giữa các cửa sổ.
- **Đầu ra của `stft()`**:
  - **`f`**: Mảng các tần số (Hz).
  - **`t`**: Mảng các thời điểm (thời gian) tại mỗi khung.
  - **`Zxx`**: Kết quả STFT, chứa giá trị biên độ (magnitude) của biến đổi Fourier tại các điểm thời gian và tần số.

```python
    return f, t, Zxx
```

- Trả về ba giá trị:
  - **`f`**: Mảng tần số.
  - **`t`**: Mảng thời gian.
  - **`Zxx`**: Kết quả STFT, chứa dữ liệu trong miền thời gian-tần số.

### 3. **Hàm `plot_stft`**:

```python
def plot_stft(f, t, Zxx):
    """
    Hàm để vẽ biểu đồ STFT (Time-Frequency Spectrogram).

    Args:
    - f: Tần số
    - t: Thời gian
    - Zxx: Kết quả STFT
    """
```

- Hàm `plot_stft` vẽ **spectrogram** (biểu đồ phổ tần số theo thời gian) từ kết quả STFT.

#### Vẽ đồ thị:

```python
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, np.abs(Zxx), shading='auto')  # Vẽ phổ tần số theo thời gian
```

- **`plt.figure(figsize=(10, 6))`**: Tạo một cửa sổ đồ họa với kích thước 10x6 inch.
- **`plt.pcolormesh(t, f, np.abs(Zxx), shading='auto')`**: Vẽ đồ thị phổ tần số theo thời gian.
  - **`t`**: Trục thời gian.
  - **`f`**: Trục tần số.
  - **`np.abs(Zxx)`**: Lấy giá trị tuyệt đối của `Zxx`, vì `Zxx` là một số phức, và chúng ta quan tâm đến biên độ (magnitude).
  - **`shading='auto'`**: Làm mịn các ô trong đồ thị.

```python
    plt.title('STFT of Audio Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(label='Magnitude')
    plt.tight_layout()
    plt.show()
```

- **`plt.title('STFT of Audio Signal')`**: Tiêu đề cho đồ thị.
- **`plt.xlabel('Time (s)')`** và **`plt.ylabel('Frequency (Hz)')`**: Gán nhãn cho các trục thời gian và tần số.
- **`plt.colorbar(label='Magnitude')`**: Thêm thanh màu hiển thị giá trị biên độ.
- **`plt.tight_layout()`**: Tự động điều chỉnh các yếu tố đồ họa để không bị chồng chéo.
- **`plt.show()`**: Hiển thị đồ thị.

### Đầu ra:

- **Biểu đồ STFT**: Đây là một **spectrogram** thể hiện mức độ thay đổi của tín hiệu theo thời gian và tần số. Các vùng sáng hơn trên biểu đồ biểu thị sự hiện diện mạnh mẽ của một tần số tại một thời điểm cụ thể.
