Giải thích chi tiết từng bước và từng dòng mã trong chương trình tính toán **Fast Fourier Transform (FFT)**, cùng với cách dữ liệu đầu vào được xử lý qua các thuật toán và đầu ra của mỗi bước.

### 1. **Định nghĩa hàm `fft_1d`** (Tính toán FFT 1D):

```python
def fft_1d(x):
    N = len(x)
    if N <= 1:
        return x
```

- Hàm `fft_1d` tính toán **Fast Fourier Transform (FFT)** cho tín hiệu 1 chiều `x`.
- **`N = len(x)`**: Lấy độ dài của tín hiệu `x`, đó là số mẫu của tín hiệu.
- **`if N <= 1:`**: Nếu độ dài của tín hiệu chỉ có một phần tử (hoặc không có tín hiệu), hàm trả về chính tín hiệu đó vì không cần phải tính toán FFT cho một tín hiệu có độ dài bằng 1.

#### 1.1 **Chia tín hiệu thành các phần chẵn và lẻ**:

```python
    even = fft_1d(x[::2])
    odd = fft_1d(x[1::2])
```

- Để tính toán FFT, thuật toán **Cooley-Tukey FFT** chia tín hiệu thành hai phần:
  - **`x[::2]`**: Tạo mảng chứa các giá trị tại các chỉ số chẵn của tín hiệu.
  - **`x[1::2]`**: Tạo mảng chứa các giá trị tại các chỉ số lẻ của tín hiệu.
- Hàm gọi đệ quy `fft_1d` để tính FFT cho phần chẵn (`even`) và phần lẻ (`odd`). Các phần này sẽ được kết hợp lại sau.

#### 1.2 **Tính toán các giá trị DFT**:

```python
    factor = np.exp(-2j * np.pi * np.arange(N) / N)
```

- **`factor`** là một mảng các yếu tố phức (complex exponential) trong công thức DFT:
  - **`np.exp(-2j * np.pi * np.arange(N) / N)`**: Tính toán yếu tố phức sử dụng công thức DFT, nơi:
    - `2j * np.pi` là tham số liên quan đến chu kỳ (vì DFT sử dụng các số phức có dạng \( e^{-j2\pi k/N} \)).
    - **`np.arange(N)`**: Tạo mảng từ 0 đến N-1, đại diện cho các chỉ số tần số.
    - **`/ N`**: Chia cho N để chuẩn hóa các yếu tố.

#### 1.3 **Kết hợp kết quả lại**:

```python
    X = np.zeros(N, dtype=complex)
    for k in range(N // 2):
        X[k] = even[k] + factor[k] * odd[k]
        X[k + N // 2] = even[k] - factor[k] * odd[k]

    return X
```

- **`X = np.zeros(N, dtype=complex)`**: Tạo mảng `X` với kích thước `N` và kiểu dữ liệu phức (`complex`).
- Trong vòng lặp `for k in range(N // 2):`, kết quả của phần chẵn (`even[k]`) và phần lẻ (`odd[k]`) được kết hợp lại với các yếu tố phức (`factor[k]`):
  - **`X[k] = even[k] + factor[k] * odd[k]`**: Kết hợp giá trị của phần chẵn và phần lẻ.
  - **`X[k + N // 2] = even[k] - factor[k] * odd[k]`**: Cộng thêm một phần với dấu trừ cho các chỉ số tần số phía sau.
- Kết quả cuối cùng của thuật toán là mảng `X` chứa các giá trị DFT của tín hiệu `x`.

### 2. **Tải tín hiệu âm thanh và thực hiện FFT**:

```python
audio_path = f'E:\\Queenless\\20k_audio_splitted_dataset\\test\\NonQueen\\queenless_1.wav'
signal, rate = librosa.load(audio_path, sr=None)
```

- **`librosa.load(audio_path, sr=None)`**: Tải tín hiệu âm thanh từ tệp `.wav` tại `audio_path`.
  - **`sr=None`**: Đảm bảo giữ nguyên tần số lấy mẫu (sample rate) gốc của tín hiệu âm thanh.

```python
signal_part = signal[:1024]
```

- **`signal_part = signal[:1024]`**: Lấy một phần nhỏ của tín hiệu (1024 mẫu đầu tiên). Đây là tín hiệu con mà bạn sẽ tính toán FFT.

### 3. **Áp dụng FFT 1D**:

```python
X = fft_1d(signal_part)
```

- **`fft_1d(signal_part)`**: Áp dụng hàm `fft_1d` để tính toán FFT của phần tín hiệu con. Kết quả trả về là mảng `X` chứa các giá trị DFT của tín hiệu.

### 4. **Tính toán phổ tần số (Magnitude)**:

```python
magnitude = np.abs(X)
```

- **`np.abs(X)`**: Lấy giá trị **biên độ (magnitude)** của các giá trị trong mảng `X`. Bởi vì `X` chứa các số phức, ta chỉ quan tâm đến biên độ (magnitude), giúp biểu diễn cường độ tín hiệu ở mỗi tần số.

### 5. **Tính tần số tương ứng với các giá trị FFT**:

```python
frequencies = np.fft.fftfreq(len(signal_part), d=1/rate)
```

- **`np.fft.fftfreq(len(signal_part), d=1/rate)`**: Tính tần số tương ứng với các chỉ số FFT.
  - **`len(signal_part)`**: Số lượng mẫu trong phần tín hiệu con.
  - **`d=1/rate`**: Khoảng cách giữa các mẫu tín hiệu, được tính từ tỷ lệ lấy mẫu (`rate`), nghĩa là mỗi mẫu cách nhau `1/rate` giây.

### 6. **Vẽ biểu đồ phổ tần số**:

```python
plt.figure(figsize=(10, 6))
plt.plot(frequencies[:len(frequencies)//2], magnitude[:len(magnitude)//2])
plt.title('Audio Spectrum (FFT 1D)', fontsize=14)
plt.xlabel('Frequency (Hz)', fontsize=12)
plt.ylabel('Magnitude', fontsize=12)
plt.grid(True)
plt.show()
```

- **`plt.figure(figsize=(10, 6))`**: Tạo một cửa sổ đồ họa với kích thước 10x6 inch.
- **`plt.plot(frequencies[:len(frequencies)//2], magnitude[:len(magnitude)//2])`**: Vẽ biểu đồ phổ tần số chỉ với phần tần số dương (do phổ DFT là đối xứng).
- **`plt.title`, `plt.xlabel`, `plt.ylabel`**: Đặt tiêu đề và nhãn cho trục.
- **`plt.grid(True)`**: Thêm lưới vào đồ thị để dễ đọc hơn.
- **`plt.show()`**: Hiển thị đồ thị.

### Đầu ra của từng bước:

- **`X`**: Mảng chứa kết quả DFT của tín hiệu.
- **`magnitude`**: Biên độ của DFT, cho biết cường độ tín hiệu tại các tần số.
- **`frequencies`**: Mảng chứa các giá trị tần số tương ứng với mỗi giá trị trong DFT.
- **Biểu đồ phổ tần số**: Biểu đồ thể hiện cường độ của tín hiệu tại các tần số dương, giúp phân tích các đặc tính tần số của tín hiệu âm thanh.

### Tóm tắt:

Chương trình này sử dụng thuật toán **FFT 1D** để tính toán phổ tần số của tín hiệu âm thanh. Bằng cách chia tín hiệu thành các phần chẵn và lẻ, thuật toán đệ quy tính toán FFT và kết hợp kết quả lại để có được phổ tần số của tín hiệu. Biểu đồ phổ tần số được vẽ để giúp phân tích các đặc tính tần số của tín hiệu âm thanh.
