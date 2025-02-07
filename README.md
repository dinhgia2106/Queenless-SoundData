# I. Chiết xuất đặc trưng

---

### 1. **Mel-frequency Cepstral Coefficients (MFCCs)**

#### Mục đích và cách hoạt động:

- **MFCCs** là một trong những kỹ thuật phổ biến nhất để trích xuất các đặc trưng âm thanh quan trọng từ tín hiệu âm thanh. Trong dự án xác định sự có mặt của ong chúa, **MFCCs** sẽ giúp phân tích các âm thanh trong tổ ong (như tiếng rít của ong chúa) và chuyển chúng thành các đặc trưng dễ dàng nhận dạng.

#### Các bước thực hiện:

1. **Pre-emphasis**: Áp dụng bộ lọc high-pass để làm nổi bật các tần số cao, giúp tín hiệu dễ phân biệt hơn trong môi trường ồn ào.
2. **Framing**: Chia tín hiệu thành các khung nhỏ (thường là khoảng vài miligiây), vì tín hiệu âm thanh thay đổi theo thời gian.
3. **Windowing**: Nhân mỗi khung với một hàm cửa sổ (ví dụ: Hamming) để giảm sự biến động ở biên của khung.
4. **FFT (Fast Fourier Transform)**: Tính toán phổ tần số của mỗi khung để chuyển tín hiệu từ miền thời gian sang miền tần số.
5. **Mel Filterbank**: Áp dụng một bộ lọc Mel để phân chia phổ thành các dải tần số có mức độ giống cách mà tai người cảm nhận âm thanh.
6. **Logarithm**: Áp dụng phép toán logarithm để làm nổi bật những tần số quan trọng.
7. **Discrete Cosine Transform (DCT)**: Sử dụng DCT để giảm số lượng đặc trưng và giữ lại thông tin quan trọng, tạo ra các hệ số MFCC.

#### Ứng dụng trong dự án:

- MFCC giúp phân tích đặc trưng âm thanh, từ đó phân biệt tiếng rít của ong chúa với các âm thanh khác trong tổ ong.

---

### 2. **Short-time Fourier Transform (STFT)**

#### Mục đích và cách hoạt động:

- **STFT** là một phương pháp để phân tích tín hiệu âm thanh trong cả miền thời gian và tần số. Nó chia tín hiệu thành các khung nhỏ, sau đó áp dụng Fourier Transform để chuyển mỗi khung từ miền thời gian sang miền tần số.

#### Các bước thực hiện:

1. **Chia tín hiệu thành các khung nhỏ**: Tín hiệu âm thanh của ong sẽ được chia thành các đoạn nhỏ (ví dụ, 20-40ms) để có thể phân tích biến động trong âm thanh.
2. **Áp dụng Fourier Transform**: Tính toán Fourier Transform cho mỗi khung để phân tích các tần số trong tín hiệu.
3. **Thời gian và tần số**: Sau khi tính toán STFT, ta có thể hình dung được các tần số của tín hiệu trong mỗi khoảng thời gian. Điều này rất hữu ích để phát hiện các thay đổi tần số trong tiếng ong chúa.

#### Ứng dụng trong dự án:

- **STFT** giúp phân tích sự thay đổi của các thành phần tần số trong tiếng của ong, từ đó giúp nhận diện âm thanh của ong chúa qua các dải tần số đặc trưng.

---

### 3. **Fast Fourier Transform (FFT)**

#### Mục đích và cách hoạt động:

- **FFT** là một thuật toán nhanh chóng để tính toán **Discrete Fourier Transform (DFT)**, chuyển tín hiệu từ miền thời gian sang miền tần số. Với **FFT**, ta có thể phân tích phổ tần số của tín hiệu âm thanh nhanh chóng hơn so với phương pháp DFT thông thường.

#### Các bước thực hiện:

1. **Phân tích phổ**: Tín hiệu âm thanh của ong chúa sẽ được chuyển đổi từ miền thời gian sang miền tần số thông qua **FFT**.
2. **Tần số và biên độ**: Sau khi tính toán FFT, ta có thể xác định được các thành phần tần số của tín hiệu và biên độ của chúng.

#### Ứng dụng trong dự án:

- **FFT** giúp nhanh chóng phân tích các thành phần tần số trong âm thanh của tổ ong và xác định các đặc trưng tần số quan trọng, đặc biệt là các tần số đặc trưng của tiếng rít của ong chúa.

---

### 4. **Constant-Q Transform (CQT)**

#### Mục đích và cách hoạt động:

- **CQT** là một phương pháp phân tích phổ tần số mà các dải tần số được phân chia theo tỷ lệ logarithmic, giúp biểu diễn chính xác các cao độ và hài hòa của âm thanh, phù hợp với cách con người nghe nhạc. Với **CQT**, các dải tần số thấp sẽ rộng hơn và các dải tần số cao sẽ hẹp hơn, giúp phát hiện các đặc trưng âm thanh của ong chúa.

#### Các bước thực hiện:

1. **Chia tín hiệu thành các dải tần số**: **CQT** chia tín hiệu thành các dải tần số logarithmic.
2. **Phân tích tần số**: CQT tính toán các thành phần tần số trong tín hiệu theo tỷ lệ logarithmic, giúp phân tích rõ hơn các tần số thấp và cao trong tiếng ong chúa.

#### Ứng dụng trong dự án:

- **CQT** giúp phân tích tiếng rít của ong chúa với các tần số hài hòa chính xác hơn, đặc biệt là trong các âm thanh nhạc cụ hoặc âm thanh động vật có cao độ chính xác.

---

### 5. **Spectral Contrast (SC)**

#### Mục đích và cách hoạt động:

- **SC** đo lường sự khác biệt giữa các đỉnh và đáy trong phổ tần số. Nó giúp phân biệt giữa các loại âm thanh bằng cách so sánh năng lượng giữa các dải tần số lân cận.

#### Các bước thực hiện:

1. **Phân chia phổ tần số**: Chia tín hiệu thành các dải tần số và tính toán năng lượng trong mỗi dải.
2. **Tính SC**: Tính sự khác biệt giữa năng lượng của các dải tần số liền kề.

#### Ứng dụng trong dự án:

- **SC** giúp phân biệt tiếng rít của ong chúa với các âm thanh khác trong tổ ong, vì tiếng rít của ong chúa có sự tương phản rõ rệt trong phổ tần số.

---

### 6. **Chroma**

#### Mục đích và cách hoạt động:

- **Chroma** giúp đại diện cho nội dung hài hòa và đặc tính âm sắc của tín hiệu âm thanh. Với **Chroma**, tín hiệu âm thanh sẽ được ánh xạ vào các lớp âm (pitch classes), giúp phân tích các nốt nhạc trong âm thanh. Trong dự án xác định sự có mặt của ong chúa, **Chroma** giúp nhận diện các đặc trưng âm sắc của tiếng rít từ ong chúa.

#### Các bước thực hiện:

1. **Phân tích phổ tần số**: Sử dụng **STFT** hoặc các phương pháp tương tự để phân tích tín hiệu.
2. **Ánh xạ tần số thành các lớp âm**: Chuyển tần số vào các lớp âm (ví dụ, C, C#, D, ...).
3. **Tính toán Chroma**: Tính toán đặc trưng **Chroma** của tín hiệu từ các lớp âm.

#### Ứng dụng trong dự án:

- **Chroma** giúp phân tích tiếng rít của ong chúa trong môi trường tổ ong, xác định các nốt nhạc hoặc âm sắc đặc trưng của âm thanh này.

---

### Tóm tắt:

- **MFCCs**, **STFT**, **FFT**, **CQT**, **SC**, và **Chroma** là các phương pháp phân tích tín hiệu âm thanh cực kỳ mạnh mẽ, giúp trích xuất đặc trưng âm thanh từ tín hiệu trong việc xác định sự có mặt của ong chúa trong tổ ong.
- Các phương pháp này giúp phân tích và trích xuất thông tin về tần số, cao độ, sự tương phản âm sắc, và các đặc trưng âm thanh khác từ tiếng rít của ong chúa, giúp hệ thống nhận diện được âm thanh đặc trưng của ong chúa so với các âm thanh khác trong môi trường tổ ong.
- Việc sử dụng các phương pháp này giúp phân biệt rõ ràng tiếng rít của ong chúa, một đặc trưng âm thanh quan trọng trong dự án phân loại âm thanh này.
#   Q u e e n l e s s - S o u n d D a t a  
 