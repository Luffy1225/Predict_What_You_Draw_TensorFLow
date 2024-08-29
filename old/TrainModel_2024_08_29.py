import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tensorflow.keras.datasets import mnist # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense # type: ignore

from old.Create_train_data_2024_08_28 import load_train_data


# 載入數據
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
(train_images, train_labels), (test_images, test_labels) = load_train_data()  # 預計接口
# train_images train_labels test_images test_labels 都是 numpy.ndarray



# 數據預處理
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


label_amount = len(train_labels[0])






# 假設這是你的模型定義和訓練代碼
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(label_amount, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 訓練模型
Epochs = 25

print(train_images.shape)
print(train_labels.shape)


history = model.fit(train_images, train_labels, epochs=Epochs, batch_size=64, validation_split=0.2)


models_path = "models"

model_name = "Luffy_Made"

# 保存模型
model_full_name = f'{model_name}_Epoch_{Epochs}.h5'

model_full_name = os.path.join(models_path, model_full_name)

model.save(model_full_name)