import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix

# Tải & tiền xử lý dữ liệu
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Chuẩn hóa dữ liệu (đưa về giá trị từ 0 đến 1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Thêm chiều kênh (1 kênh grayscale)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# One-hot encode cho nhãn (giúp chuyển nhãn từ dạng số nguyên (0-9) sang vector one-hot để phù hợp với cách mô hình CNN hoạt động)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

#Xây dựng mô hình CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

#Biên dịch và huấn luyện mô hình
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4. Huấn luyện mô hình và lưu lịch sử
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Đánh giá mô hình trên tập kiểm tra
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Accuracy: {test_acc:.2%}")
print(f"Loss: {test_loss:.2%}")

#Trực quan hóa kết quả:
plt.figure(figsize=(12, 5))

# Độ chính xác
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Quá trình cải thiện độ chính xác')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Mất mát
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Quá trình giảm thiểu mất mát')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Hiển thị biểu đồ
plt.tight_layout()
plt.show()

# Tính toán độ chính xác cho từng class
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Chọn nhãn có xác suất cao nhất
y_true = np.argmax(y_test, axis=1)  # Nhãn thực tế

classes = np.unique(y_true)
accuracy_per_class = []
samples_per_class = []

for cls in classes:
    cls_idx = np.where(y_true == cls)[0]  # Lấy index của các mẫu thuộc lớp `cls`
    correct_predictions = np.sum(y_pred_classes[cls_idx] == cls)
    accuracy = correct_predictions / len(cls_idx)
    accuracy_per_class.append(accuracy)
    samples_per_class.append(len(cls_idx))

# Tạo bảng liệt kê
df = pd.DataFrame({
    "Class": classes,
    "Accuracy": accuracy_per_class,
    "Samples": samples_per_class
})

# Vẽ bảng liệt kê
fig, ax = plt.subplots(figsize=(6, 3))
ax.axis('off')  # Ẩn khung vẽ
table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(df.columns))))

# Vẽ ma trận nhầm lẫn
conf_matrix = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix')
plt.xlabel('Prediction')
plt.ylabel('Class')
plt.show()

# Hiển thị 5 hình ảnh và dự đoán của mô hình
predictions = model.predict(x_test)

for i in range(5):
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Dự đoán: {np.argmax(predictions[i])}")
    plt.axis('off')
    plt.show()

# 10. Lưu mô hình (Optional)
model.save('my_model.keras')
