# Predict_What_You_Draw

- [English Version Instructions](/README.md)

這是一個基於 Tkinter 和 PIL 庫的手繪圖像識別應用程式。用戶可以在應用程式中手繪數字或 (英文)，並使用訓練好的 MNIST 模型 或是自行訓練 的模型 進行預測。

# Install 

```
git clone https://github.com/Luffy1225/Predict_What_You_Draw_TensorFLow.git
cd Predict_What_You_Draw_TensorFLow
pip install -r requirements.txt
```


# Run

> [!IMPORTANT]
> 執行前先確認 `images` 資料夾存在 (否則找不到對應訓練集)
> 可解壓縮 附帶圖片 images_DATE 並命名為 `images`

## 開啟 Predict What You Draw

```
python main.py
```

## 開啟 AI 訓練
```
python AIModel.py
```

## 開啟 檔案管理系統
```
python FolderManager.py
```



## Predict What You Draw 功能

1. 手繪畫布：

    用戶可以在應用程式提供的畫布上自由繪製數字。
1. 模型選擇：

    提供多個訓練好的模型，讓用戶選擇並切換進行預測。
    
    > [!NOTE]
    > 推薦 使用 `mnist_model_Epoch_100.keras` 模型
    

1. 預測結果顯示：

    應用程式會在畫布下方顯示模型預測的數字。

2. 清除畫布：

    用戶可以清除畫布上的所有內容，重新繪製數字。

3. 保存圖像：

    可以將繪製的圖像保存到本機，以作做為訓練集。


## 如何訓練自己的模型？

- [tutorial](/tutorial_zh_TW.md)


## `images` 資料夾 結構

    images/
    ├── ignore/                 # 忽略的資料，不用於訓練或測試
    │   ├── label_1/            # 標籤 1 的圖片資料
    │   │   ├── 0.jpg           # 圖片檔案
    │   │   ├── 1.jpg
    │   │   └── ...             # 更多圖片
    │   ├── label_2/            # 標籤 2 的圖片資料
    │   │   ├── 0.jpg
    │   │   ├── 1.jpg
    │   │   └── ...
    │   └── label_x/            # 其他標籤的圖片資料
    │       ├── 0.jpg
    │       ├── 1.jpg
    │       └── ...
    ├── test/                   # 測試數據，用於驗證模型
    │   ├── label_1/            # 標籤 1 的測試圖片
    │   │   ├── 0.jpg
    │   │   ├── 1.jpg
    │   │   └── ...
    │   ├── label_2/            # 標籤 2 的測試圖片
    │   │   ├── 0.jpg
    │   │   ├── 1.jpg
    │   │   └── ...
    │   └── label_x/            # 其他標籤的測試圖片
    │       ├── 0.jpg
    │       ├── 1.jpg
    │       └── ...
    ├── train/                  # 訓練數據，用於訓練模型
    │   ├── label_1/            # 標籤 1 的訓練圖片
    │   │   ├── 0.jpg
    │   │   ├── 1.jpg
    │   │   └── ...
    │   ├── label_2/            # 標籤 2 的訓練圖片
    │   │   ├── 0.jpg
    │   │   ├── 1.jpg
    │   │   └── ...
    │   └── label_x/            # 其他標籤的訓練圖片
    │       ├── 0.jpg
    │       ├── 1.jpg
    │       └── ...


    
