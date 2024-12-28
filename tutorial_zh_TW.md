- [返回說明](README_zh_TW.md)

# 怎麼訓練自己的模型？

1. 確保 `images` 、`images/test` 和 `images/train` 資料夾存在（你可以在 `FolderManager.py` 中生成這些資料夾結構）。
1. 將你的資料集放入對應`test`跟`train`的資料夾中。
1. 檢查你的資料集的標籤（`images/test` 和 `images/train` 中的資料夾名稱即為資料集的標籤）。
1. 執行`AIModel.py` 並 開始訓練。
1. 享受你的專屬模型吧(你可以在`main.py`測試看看你自己的模型)！

# 怎麼建立自己的資料集？

1. 確保 images 、images/test 和 images/train 資料夾已經存在（你可以在 FolderManager.py 中生成這些資料夾結構）。
1. 執行 main.py。
1. 在畫布上繪製圖片。
2. 在 Correction 區域輸入標籤（區域內的文字即為你的訓練標籤）。
3. 選擇儲存資料夾（如果你希望這張圖片作為測試/訓練資料集的一部分）。
4. 右鍵點擊保存當前圖片。
5. 重複步驟 3 ~ 6，這樣你就可以建立自己的資料集啦！