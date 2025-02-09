# I. Chiết xuất đặc trưng

---

## 1. **Mel-frequency Cepstral Coefficients (MFCCs)**

### Mục đích và cách hoạt động:

- **MFCCs** là một trong những kỹ thuật phổ biến nhất để trích xuất các đặc trưng âm thanh quan trọng từ tín hiệu âm thanh. Trong dự án xác định sự có mặt của ong chúa, **MFCCs** sẽ giúp phân tích các âm thanh trong tổ ong (như tiếng rít của ong chúa) và chuyển chúng thành các đặc trưng dễ dàng nhận dạng.

### Các bước thực hiện:

1. **Pre-emphasis**: Áp dụng bộ lọc high-pass để làm nổi bật các tần số cao, giúp tín hiệu dễ phân biệt hơn trong môi trường ồn ào.
2. **Framing**: Chia tín hiệu thành các khung nhỏ (thường là khoảng vài miligiây), vì tín hiệu âm thanh thay đổi theo thời gian.
3. **Windowing**: Nhân mỗi khung với một hàm cửa sổ (ví dụ: Hamming) để giảm sự biến động ở biên của khung.
4. **FFT (Fast Fourier Transform)**: Tính toán phổ tần số của mỗi khung để chuyển tín hiệu từ miền thời gian sang miền tần số.
5. **Mel Filterbank**: Áp dụng một bộ lọc Mel để phân chia phổ thành các dải tần số có mức độ giống cách mà tai người cảm nhận âm thanh.
6. **Logarithm**: Áp dụng phép toán logarithm để làm nổi bật những tần số quan trọng.
7. **Discrete Cosine Transform (DCT)**: Sử dụng DCT để giảm số lượng đặc trưng và giữ lại thông tin quan trọng, tạo ra các hệ số MFCC.

### Ứng dụng trong dự án:

- MFCC giúp phân tích đặc trưng âm thanh, từ đó phân biệt tiếng rít của ong chúa với các âm thanh khác trong tổ ong.

---

## 2. **Short-time Fourier Transform (STFT)**

### Mục đích và cách hoạt động:

- **STFT** là một phương pháp để phân tích tín hiệu âm thanh trong cả miền thời gian và tần số. Nó chia tín hiệu thành các khung nhỏ, sau đó áp dụng Fourier Transform để chuyển mỗi khung từ miền thời gian sang miền tần số.

### Các bước thực hiện:

1. **Chia tín hiệu thành các khung nhỏ**: Tín hiệu âm thanh của ong sẽ được chia thành các đoạn nhỏ (ví dụ, 20-40ms) để có thể phân tích biến động trong âm thanh.
2. **Áp dụng Fourier Transform**: Tính toán Fourier Transform cho mỗi khung để phân tích các tần số trong tín hiệu.
3. **Thời gian và tần số**: Sau khi tính toán STFT, ta có thể hình dung được các tần số của tín hiệu trong mỗi khoảng thời gian. Điều này rất hữu ích để phát hiện các thay đổi tần số trong tiếng ong chúa.

### Ứng dụng trong dự án:

- **STFT** giúp phân tích sự thay đổi của các thành phần tần số trong tiếng của ong, từ đó giúp nhận diện âm thanh của ong chúa qua các dải tần số đặc trưng.

---

## 3. **Fast Fourier Transform (FFT)**

### Mục đích và cách hoạt động:

- **FFT** là một thuật toán nhanh chóng để tính toán **Discrete Fourier Transform (DFT)**, chuyển tín hiệu từ miền thời gian sang miền tần số. Với **FFT**, ta có thể phân tích phổ tần số của tín hiệu âm thanh nhanh chóng hơn so với phương pháp DFT thông thường.

### Các bước thực hiện:

1. **Phân tích phổ**: Tín hiệu âm thanh của ong chúa sẽ được chuyển đổi từ miền thời gian sang miền tần số thông qua **FFT**.
2. **Tần số và biên độ**: Sau khi tính toán FFT, ta có thể xác định được các thành phần tần số của tín hiệu và biên độ của chúng.

### Ứng dụng trong dự án:

- **FFT** giúp nhanh chóng phân tích các thành phần tần số trong âm thanh của tổ ong và xác định các đặc trưng tần số quan trọng, đặc biệt là các tần số đặc trưng của tiếng rít của ong chúa.

---

## 4. **Constant-Q Transform (CQT)**

### Mục đích và cách hoạt động:

- **CQT** là một phương pháp phân tích phổ tần số mà các dải tần số được phân chia theo tỷ lệ logarithmic, giúp biểu diễn chính xác các cao độ và hài hòa của âm thanh, phù hợp với cách con người nghe nhạc. Với **CQT**, các dải tần số thấp sẽ rộng hơn và các dải tần số cao sẽ hẹp hơn, giúp phát hiện các đặc trưng âm thanh của ong chúa.

### Các bước thực hiện:

1. **Chia tín hiệu thành các dải tần số**: **CQT** chia tín hiệu thành các dải tần số logarithmic.
2. **Phân tích tần số**: CQT tính toán các thành phần tần số trong tín hiệu theo tỷ lệ logarithmic, giúp phân tích rõ hơn các tần số thấp và cao trong tiếng ong chúa.

### Ứng dụng trong dự án:

- **CQT** giúp phân tích tiếng rít của ong chúa với các tần số hài hòa chính xác hơn, đặc biệt là trong các âm thanh nhạc cụ hoặc âm thanh động vật có cao độ chính xác.

---

## 5. **Spectral Contrast (SC)**

### Mục đích và cách hoạt động:

- **SC** đo lường sự khác biệt giữa các đỉnh và đáy trong phổ tần số. Nó giúp phân biệt giữa các loại âm thanh bằng cách so sánh năng lượng giữa các dải tần số lân cận.

### Các bước thực hiện:

1. **Phân chia phổ tần số**: Chia tín hiệu thành các dải tần số và tính toán năng lượng trong mỗi dải.
2. **Tính SC**: Tính sự khác biệt giữa năng lượng của các dải tần số liền kề.

### Ứng dụng trong dự án:

- **SC** giúp phân biệt tiếng rít của ong chúa với các âm thanh khác trong tổ ong, vì tiếng rít của ong chúa có sự tương phản rõ rệt trong phổ tần số.

---

## 6. **Chroma**

### Mục đích và cách hoạt động:

- **Chroma** giúp đại diện cho nội dung hài hòa và đặc tính âm sắc của tín hiệu âm thanh. Với **Chroma**, tín hiệu âm thanh sẽ được ánh xạ vào các lớp âm (pitch classes), giúp phân tích các nốt nhạc trong âm thanh. Trong dự án xác định sự có mặt của ong chúa, **Chroma** giúp nhận diện các đặc trưng âm sắc của tiếng rít từ ong chúa.

### Các bước thực hiện:

1. **Phân tích phổ tần số**: Sử dụng **STFT** hoặc các phương pháp tương tự để phân tích tín hiệu.
2. **Ánh xạ tần số thành các lớp âm**: Chuyển tần số vào các lớp âm (ví dụ, C, C#, D, ...).
3. **Tính toán Chroma**: Tính toán đặc trưng **Chroma** của tín hiệu từ các lớp âm.

### Ứng dụng trong dự án:

- **Chroma** giúp phân tích tiếng rít của ong chúa trong môi trường tổ ong, xác định các nốt nhạc hoặc âm sắc đặc trưng của âm thanh này.

---

## Tóm tắt:

- **MFCCs**, **STFT**, **FFT**, **CQT**, **SC**, và **Chroma** là các phương pháp phân tích tín hiệu âm thanh cực kỳ mạnh mẽ, giúp trích xuất đặc trưng âm thanh từ tín hiệu trong việc xác định sự có mặt của ong chúa trong tổ ong.
- Các phương pháp này giúp phân tích và trích xuất thông tin về tần số, cao độ, sự tương phản âm sắc, và các đặc trưng âm thanh khác từ tiếng rít của ong chúa, giúp hệ thống nhận diện được âm thanh đặc trưng của ong chúa so với các âm thanh khác trong môi trường tổ ong.
- Việc sử dụng các phương pháp này giúp phân biệt rõ ràng tiếng rít của ong chúa, một đặc trưng âm thanh quan trọng trong dự án phân loại âm thanh này.

### **Ứng dụng trong bài toán xác định ong chúa**

- Các phương pháp trên được sử dụng để trích xuất đặc trưng từ tín hiệu âm thanh của tổ ong. Sau khi trích xuất đặc trưng, các thuật toán học máy (machine learning) như **K-Nearest Neighbors (KNN)**, **Support Vector Machine (SVM)**, **Random Forest (RF)**, **Extra Trees (ET)**, và **Logistic Regression (LR)** được áp dụng để phân loại trạng thái của tổ ong, xác định xem có sự hiện diện của ong chúa hay không.

### **Quy trình tổng quát**:

1. **Thu thập tín hiệu âm thanh**: Ghi lại âm thanh trong tổ ong (có thể là tiếng rít của ong chúa).
2. **Trích xuất đặc trưng**: Sử dụng các phương pháp như **MFCCs**, **STFT**, **FFT**, **CQT**, **SC**, và **Chroma** để trích xuất các đặc trưng quan trọng từ tín hiệu.
3. **Huấn luyện mô hình**: Cung cấp các đặc trưng đã trích xuất cho các thuật toán học máy để huấn luyện mô hình phân loại.
4. **Kiểm tra mô hình**: Áp dụng mô hình đã huấn luyện vào các dữ liệu mới để xác định sự có mặt của ong chúa dựa trên âm thanh.

# II. Các Phương Pháp Học Máy (Machine Learning)

Các thuật toán học máy được sử dụng nhằm phân loại các đặc trưng âm thanh được trích xuất (MFCCs, STFT, FFT, CQT, SC, Chroma) từ tiếng rít của ong chúa. Dưới đây là mô tả chi tiết cho từng phương pháp:

---

## 1. **K-Nearest Neighbors (KNN)**

### Mục đích và cách hoạt động:

- **KNN** là một thuật toán dựa trên khoảng cách trong không gian đặc trưng. Giả định rằng các mẫu có đặc trưng tương tự sẽ nằm gần nhau, nên nhãn của mẫu mới được quyết định dựa trên nhãn của các mẫu gần nhất.
- Trong dự án, sau khi trích xuất các đặc trưng âm thanh, KNN sẽ so sánh khoảng cách giữa vector đặc trưng của mẫu chưa biết với các mẫu đã có nhãn trong tập huấn luyện. Nhãn của mẫu mới được xác định theo đa số nhãn của \( k \) láng giềng gần nhất.

### Ưu điểm:

- Dễ triển khai và trực quan.
- Không cần giai đoạn huấn luyện phức tạp (trường hợp chỉ có tính toán khoảng cách).

### Nhược điểm:

- Hiệu năng giảm đáng kể với dữ liệu có số lượng mẫu lớn vì tính toán khoảng cách cho tất cả các cặp mẫu.
- Kết quả phụ thuộc mạnh vào lựa chọn số lượng láng giềng \( k \) và cách xác định khoảng cách.

---

## 2. **Support Vector Machine (SVM)**

### Mục đích và cách hoạt động:

- **SVM** tìm kiếm một siêu phẳng tối ưu phân chia các lớp trong không gian đặc trưng. Với các trường hợp không tuyến tính, SVM sử dụng “kernel trick” (ví dụ: kernel RBF) để chuyển đổi dữ liệu vào không gian đặc trưng cao hơn.
- Trong dự án, SVM được tối ưu hóa qua GridSearchCV với các tham số như `{'C': 10, 'gamma': 1, 'kernel': 'rbf'}` nhằm phân biệt chính xác các đặc trưng âm thanh liên quan đến tiếng rít của ong chúa.

### Ưu điểm:

- Hiệu quả trong không gian đặc trưng cao và với dữ liệu có sự phân tách không tuyến tính.
- Có khả năng tổng quát hóa tốt nếu được tối ưu tham số đúng cách.

### Nhược điểm:

- Huấn luyện SVM với kernel RBF có thể tốn thời gian, đặc biệt với tập dữ liệu lớn.
- Yêu cầu phải tối ưu hóa các tham số siêu để đạt hiệu suất tốt nhất.

---

## 3. **Logistic Regression (LR)**

### Mục đích và cách hoạt động:

- **Logistic Regression** là mô hình tuyến tính dùng cho bài toán phân loại nhị phân. Mô hình ước lượng xác suất thuộc về mỗi lớp qua hàm logistic (sigmoid) và gán nhãn cho mẫu dựa trên ngưỡng 0.5.
- Trong dự án, LR được áp dụng cho các đặc trưng đã chuẩn hóa để xác định sự có mặt của ong chúa, là một lựa chọn đơn giản nhưng hiệu quả trong nhiều trường hợp phân loại nhị phân.

### Ưu điểm:

- Dễ hiểu, triển khai nhanh và cho kết quả huấn luyện nhanh.
- Hiệu quả trong các bài toán phân loại tuyến tính.

### Nhược điểm:

- Có khả năng xử lý kém các mối quan hệ phi tuyến tính nếu dữ liệu không thể tuyến tính hóa được.
- Có thể bị giới hạn khi dữ liệu có nhiễu cao hoặc số lượng đặc trưng lớn.

---

## 4. **Random Forest (RF)**

### Mục đích và cách hoạt động:

- **Random Forest** là một thuật toán ensemble kết hợp nhiều cây quyết định (decision trees) để cải thiện độ chính xác và giảm overfitting.
- Trong dự án, RF được huấn luyện trên các đặc trưng âm thanh (có thể là dữ liệu gốc không cần chuẩn hóa) và mỗi cây được huấn luyện trên một tập con ngẫu nhiên của dữ liệu, sau đó kết hợp kết quả thông qua voting.

### Ưu điểm:

- Khả năng xử lý tốt với dữ liệu nhiễu và không bị overfitting khi số lượng cây đủ lớn.
- Mô hình có thể xử lý cả dữ liệu tuyến tính lẫn phi tuyến tính.

### Nhược điểm:

- Mô hình có thể trở nên phức tạp và khó giải thích.
- Tiêu thụ tài nguyên tính toán cao khi số lượng cây tăng.

---

## 5. **Extra Trees (ET)**

### Mục đích và cách hoạt động:

- **Extra Trees** (Extremely Randomized Trees) cũng là một thuật toán ensemble tương tự Random Forest nhưng có thêm yếu tố ngẫu nhiên trong việc chọn điểm cắt của cây, giúp giảm phương sai.
- Trong dự án, ET được sử dụng trên dữ liệu đã chuẩn hóa để phân loại các đặc trưng âm thanh. Nhờ tính chất ngẫu nhiên cao, ET thường cho kết quả ổn định và nhanh hơn trong huấn luyện.

### Ưu điểm:

- Tốc độ huấn luyện nhanh và hiệu quả giảm overfitting.
- Thường cho kết quả tốt trong nhiều bài toán phân loại.

### Nhược điểm:

- Do tính ngẫu nhiên cao, mô hình có thể cho kết quả không ổn định nếu không được hiệu chỉnh tham số cẩn thận.
- Đôi khi khó giải thích kết quả do sự phức tạp của tập hợp các cây.

---

## Tóm tắt Quy Trình Huấn Luyện và Đánh Giá ML

1. **Tiền xử lý dữ liệu**: Sau khi trích xuất các đặc trưng âm thanh (MFCCs, STFT, FFT, CQT, SC, Chroma), dữ liệu được chuyển đổi (ví dụ: từ chuỗi sang số qua LabelEncoder) và chuẩn hóa (sử dụng StandardScaler) nếu cần.
2. **Huấn luyện mô hình**: Các thuật toán ML (KNN, SVM, LR, RF, ET) được huấn luyện trên tập dữ liệu đặc trưng.
   - Ví dụ, SVM được tối ưu qua GridSearchCV với tham số `{'C': 10, 'gamma': 1, 'kernel': 'rbf'}`.
3. **Tối ưu hóa**: Các siêu tham số của mô hình được điều chỉnh (thông qua GridSearchCV hoặc RandomizedSearchCV) để đạt được hiệu suất tốt nhất.
4. **Đánh giá mô hình**: Các mô hình được đánh giá trên tập validation và test. Các chỉ số như accuracy được sử dụng để so sánh hiệu quả của từng mô hình.
5. **Lựa chọn mô hình tối ưu**: Dựa vào các chỉ số đánh giá, mô hình tốt nhất sẽ được lựa chọn để triển khai trong hệ thống nhận diện tiếng rít của ong chúa.

---

Bằng việc kết hợp các phương pháp trích xuất đặc trưng mạnh mẽ với các thuật toán học máy đã nêu, dự án không chỉ nhận diện được tiếng rít của ong chúa trong tổ ong mà còn có thể phân biệt một cách chính xác giữa các loại âm thanh trong môi trường tổ ong. Các phương pháp ML cung cấp các giải pháp đa dạng, từ các thuật toán đơn giản như KNN và LR cho đến các mô hình phức tạp và hiệu quả như SVM, RF và ET, giúp tối ưu hóa hiệu suất nhận diện trong từng trường hợp cụ thể.

---

## Kết quả

| Method | KNN    | SVM    | LR     | RF     | ET     |
| ------ | ------ | ------ | ------ | ------ | ------ |
| FFT    | 94.25% | 97.82% | 85.97% | 94.67% | 93.95% |
| STFT   | 94.50% | 56.62% | 89.50% | 94.65% | 94.10% |
| MFCC   | 92.75% | 94.00% | 73.28% | 92.00% | 92.53% |
| CQT    | 95.83% | 80.17% | 83.60% | 94.55% | 95.08% |
| CHROMA | 73.22% | 76.17% | 63.12% | 76.42% | 75.42% |
| SC     | 65.70% | 57.00% | 53.47% | 67.33% | 67.27% |
