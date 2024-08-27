
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# 載入數據
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 數據預處理
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)




# 假設這是你的模型定義和訓練代碼
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 訓練模型
Epochs = 25

history = model.fit(train_images, train_labels, epochs=Epochs, batch_size=64, validation_split=0.2)

# 保存模型
model_name = f'mnist_model_Epoch_{Epochs}.h5'


model.save(model_name)