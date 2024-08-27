import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tensorflow.keras.models import load_model
import cv2
import os

# 載入訓練好的模型


model_filename = 'mnist_model_Epoch_60.h5'  # 修改為實際模型文件名
model_folder = 'models'
model_path = os.path.join(model_folder, model_filename)

model = load_model(model_path)

# 圖片文件夾路徑
image_folder = 'image'  # 修改為實際圖片文件夾路徑

# 載入圖片並進行預處理
def load_and_preprocess_image(img_path):
    if not os.path.exists(img_path):
        print(f"圖片不存在: {img_path}")
        return None
    
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 讀取為灰度圖
    if img is None:
        print(f"無法讀取圖片: {img_path}")
        return None
    
    img = cv2.resize(img, (28, 28))  # 調整大小為 28x28
    img = img.astype('float32') / 255  # 正規化
    img = np.expand_dims(img, axis=-1)  # 增加一維以匹配模型輸入
    return img

# 預測圖片
def predict_images(image_folder):
    # 獲取資料夾中的所有檔案
    filenames = [f for f in os.listdir(image_folder) if f.endswith('.png')]
    images = []
    valid_filenames = []
    
    for filename in filenames:
        img_path = os.path.join(image_folder, filename)
        img = load_and_preprocess_image(img_path)
        if img is not None:  # 只添加有效的圖片
            images.append(img)
            valid_filenames.append(filename)
    
    if len(images) == 0:
        print("沒有有效的圖片進行預測。")
        return valid_filenames, [], []
    
    images = np.array(images)
    predictions = model.predict(images)
    predicted_labels = np.argmax(predictions, axis=1)
    
    return valid_filenames, images, predicted_labels

# 顯示圖片及預測結果
def show_predictions(filenames, images, predicted_labels):
    plt.figure(figsize=(15, 10))
    for i, filename in enumerate(filenames):
        plt.subplot(3, 5, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        plt.title(f'預測: {predicted_labels[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# 執行預測並顯示結果
filenames, images, predicted_labels = predict_images(image_folder)
show_predictions(filenames, images, predicted_labels)