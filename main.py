import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Tải tập dữ liệu MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Chuẩn hóa dữ liệu
x_train = x_train / 255.0
x_test = x_test / 255.0

# Thêm một kênh (channel) cho dữ liệu
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# One-hot encode nhãn
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

#Xây dựng mô hình CNN:
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

#Biên dịch và huấn luyện mô hình:
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

#Đánh giá và kiểm tra:
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Accuracy: {test_acc:.2%}")

# #Lưu và tải lại mô hình: (Ẩn đi nếu không muốn lưu lại kết quả train và sử dụng lần sau
# # Lưu mô hình
# model.save('mnist_cnn_model.h5')

# # # Tải lại mô hình
# loaded_model = tf.keras.models.load_model('mnist_cnn_model.h5')

#Trực quan hóa kết quả:
import matplotlib.pyplot as plt

predictions = model.predict(x_test)

# Hiển thị 5 hình ảnh và dự đoán của mô hình
for i in range(5):
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Dự đoán: {np.argmax(predictions[i])}")
    plt.axis('off')
    plt.show()


# Vẽ biểu đồ Độ chính xác
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

# Dự đoán nhãn trên tập kiểm tra
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Chọn nhãn có xác suất cao nhất
y_true = np.argmax(y_test, axis=1)  # Nhãn thực tế

# Tính toán độ chính xác cho từng class
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
plt.ylabel('Label')
plt.show()
