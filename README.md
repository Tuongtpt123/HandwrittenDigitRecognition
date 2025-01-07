# HandwrittenDigitRecognition
this repository include Handwritten Digit Recognition with CNN and Mnist dataset, using python and just for Education purpose, this project is completely 100% AI generated and edited by my own for my team =D

Ý TƯỞNG CHUNG:
  Tải và tiền xử lý dữ liệu:
    Sử dụng tập dữ liệu MNIST, chứa 60,000 hình ảnh để huấn luyện và 10,000 hình ảnh để kiểm tra.
    Chuẩn hóa dữ liệu (normalization) để tăng hiệu quả của mô hình.
    Chuyển đổi dữ liệu về định dạng phù hợp với CNN, chẳng hạn thêm một kênh (channel) nếu cần.
    
  Xây dựng mô hình CNN:
    Sử dụng các lớp convolution (tích chập) để trích xuất đặc trưng từ hình ảnh.
    Thêm các lớp pooling (lấy mẫu) để giảm kích thước dữ liệu và tăng hiệu quả tính toán.
    Cuối cùng, sử dụng các lớp fully connected (kết nối đầy đủ) để phân loại đầu ra.
    
  Huấn luyện mô hình:
    Biên dịch (compile) mô hình bằng hàm mất mát (loss function) và trình tối ưu hóa (optimizer).
    Huấn luyện mô hình với một số epoch, sử dụng dữ liệu huấn luyện và xác thực.
    
  Đánh giá và kiểm tra:
    Đánh giá mô hình trên dữ liệu kiểm tra để kiểm tra độ chính xác.
    Trực quan hóa kết quả, ví dụ hiển thị một số hình ảnh cùng với dự đoán của mô hình.
    
  Lưu và tải mô hình:
    Lưu mô hình đã huấn luyện để sử dụng lại mà không cần huấn luyện lại từ đầu.

DATA:
  1. Tập huấn luyện (Training set):
    80% của tập huấn luyện được sử dụng để huấn luyện mô hình.
    Đây là thông số được thiết lập thông qua validation_split=0.2 khi gọi model.fit().
    Với tập dữ liệu MNIST, x_train ban đầu chứa 60,000 mẫu, nên:
    48,000 mẫu (80%) được sử dụng để huấn luyện mô hình.
     
  2. Tập xác thực (Validation set):
    20% của tập huấn luyện được sử dụng để đánh giá mô hình trong quá trình huấn luyện.
    Do đó, từ tập x_train ban đầu:
    12,000 mẫu (20%) được sử dụng làm tập xác thực.
     
  4. Tập kiểm tra (Test set):
    Tập kiểm tra hoàn toàn độc lập, chứa 10,000 mẫu.
    Tập này được sử dụng để đánh giá hiệu suất cuối cùng của mô hình sau khi hoàn tất quá trình huấn luyện.

  Tóm tắt tỷ lệ phân chia:
    Tập huấn luyện: 48,000 mẫu (80% của 60,000 mẫu).
    Tập xác thực: 12,000 mẫu (20% của 60,000 mẫu).
    Tập kiểm tra: 10,000 mẫu (100% của tập kiểm tra ban đầu).
