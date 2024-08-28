import os
from PIL import Image
import numpy as np

# 主要接口
def load_train_data(): ## 仿製 (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    (train_images, train_labels) = get_train_image_and_label()
    (test_images, test_labels) = get_test_image_and_label()


    return (train_images, train_labels) ,(test_images, test_labels)

def get_train_image_and_label():
    path = os.path.join("images", "train")
    image,label = image_and_label(path)


    return (image, label)

def get_test_image_and_label():
    path = os.path.join("images", "test")
    image,label = image_and_label(path)

    return (image, label)

def image_and_label(train_folder): # 取得 images 和 label 的 base function
    # 指定訓練圖像資料夾的路徑

    # 創建一個空列表來儲存所有的圖像陣列和標籤
    image_list = []
    label_list = []

    # 遍歷資料夾中的所有類別資料夾
    for class_label, class_name in enumerate(os.listdir(train_folder)):
        class_folder = os.path.join(train_folder, class_name)
        
        if os.path.isdir(class_folder):  # 確保是資料夾
            for filename in os.listdir(class_folder):
                if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
                    # 構建完整的圖像路徑
                    image_path = os.path.join(class_folder, filename)
                    
                    # 使用 PIL 打開圖像並轉換為灰度圖像
                    img = Image.open(image_path).convert('L')
                    
                    # 將圖像轉換為 NumPy 陣列
                    img_array = np.array(img)
                    
                    # 將圖像陣列添加到列表中
                    image_list.append(img_array)
                    
                    # 將標籤（類別索引）添加到標籤列表中
                    label_list.append(class_label)

    print(f"{train_folder} 裡面有 {len(image_list)} 張圖像")
    # print(len(label_list))

    # 將圖像和標籤列表轉換為 NumPy 陣列
    images = np.array(image_list)
    labels = np.array(label_list)

    return (images, labels)


if __name__ == "__main__":
    (train_images, train_labels) ,(test_images, test_labels) = load_train_data()
    print(train_images.shape)
    print(train_labels.shape)
    print(test_images.shape)
    print(test_labels.shape)