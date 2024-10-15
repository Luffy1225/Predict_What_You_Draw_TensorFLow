# Predict_What_You_Draw

這是一個基於 Tkinter 和 PIL 庫的手繪圖像識別應用程式。用戶可以在應用程式中手繪數字或 (英文)，並使用訓練好的 MNIST 模型 或是自行訓練 的模型 進行預測。

# Install 

```
git clone https://github.com/Luffy1225/Mnist_Machine_Learning_Practice.git
cd Mnist_Machine_Learning_Practice
pip install -r requirements.txt
```


# Run

> [!IMPORTANT]
> 執行前先確認 images 資料夾存在 (否則找不到對應訓練集)
> 可解壓縮 附帶圖片 images_DATE 並命名為 images

## 開啟 Predict What You Draw

```
python main.py
```

## 開啟 檔案管理系統
```
python FolderManager.py
```

## 開啟 AI 訓練
```
python AIModel.py
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
