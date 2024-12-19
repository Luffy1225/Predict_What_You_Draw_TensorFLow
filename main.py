import tkinter as tk
from tkinter import ttk

import string

import os
from PIL import Image, ImageDraw
import random

from AIModel import AI_Model


class Predict_WhatUDraw_App:

    def init_tkinter(self, root):
        self.root = root
        self.root.title("Predict What You Draw")
        
        self.canvas_size = 280  # 放大顯示區域的尺寸
        self.image_size = 28
        
        self.QuickSave = True

        #variable for pencil
        self.prevPoint = [0,0]
        self.currentPoint = [0,0]

        
        # 初始化畫布
        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size, bg='black')
        self.canvas.grid(row=0, column=0, columnspan=4)  # 將畫布放在第一行，橫跨4列
        

        # 動作 Binding
        self.canvas.bind("<B1-Motion>", self._paint)
        self.canvas.bind("<ButtonRelease-1>", self._paint_reset)

        if (self.QuickSave):
            self.root.bind("<Button-3>", self.Save_image)
        
        self.image = Image.new('RGB', (28, 28), 'black')
        self.draw = ImageDraw.Draw(self.image)
        
        self.canvas_image = tk.PhotoImage(width=self.canvas_size, height=self.canvas_size)
        self.canvas.create_image((self.canvas_size//2, self.canvas_size//2), image=self.canvas_image, anchor=tk.CENTER)
        
        # 模型選擇下拉框
        models_list = self._get_Models_list()
        self.Model_combobox = ttk.Combobox(root, values=models_list,font=("Helvetica", 16))
        self.Model_combobox.config(width=30)
        self.Model_combobox.grid(row=1, column=0, columnspan=2,padx=30, pady=10)
        self.Model_combobox.bind("<<ComboboxSelected>>", self._Model_combobox_on_combobox_select)
        self.Model_combobox.current(0)

        # 開始預測按鈕
        self.Predict_button = tk.Button(self.root, text="Start Prediction", command=self.Predict,font=("Helvetica", 16))
        self.Predict_button.grid(row=2, column=0, padx=10)

        # 清除畫板按鈕
        self.Clear_button = tk.Button(self.root, text="Clear Canvas", command=self.Clear,font=("Helvetica", 16))
        self.Clear_button.grid(row=2, column=1, padx=10)

        # 預測結果無用標籤
        self.lb_label1 = tk.Label(root, text="Prediction: ", font=("Helvetica", 16))
        self.lb_label1.grid(row=3, column=0, columnspan=4, padx=30,pady=10, sticky='w')

        # 預測結果標籤
        self.lb_Predict = tk.Label(root, text="Not started", font=("Helvetica", 16))
        self.lb_Predict.config(width=20)
        self.lb_Predict.grid(row=3, column=1, columnspan=4, pady=5, sticky='w')

        # 訂正輸入標籤
        self.lb_Correction = tk.Label(root, text="Correction: ", font=("Helvetica", 16))
        self.lb_Correction.grid(row=4, column=0, columnspan=4, padx=30,pady=10, sticky='w')

        # 訂正輸入框
        self.E_Correction = tk.Entry(root,font=("Helvetica", 16))
        self.E_Correction.config(width=20)
        self.E_Correction.grid(row=4, column=1, columnspan=4, pady=5, sticky='w')

        # 保存圖片按鈕
        self.Save_button = tk.Button(self.root, text="Save Image", command=self.Save_image ,font=("Helvetica", 16))
        self.Save_button.grid(row=5, column=0, padx=5)


        # 保存選擇下拉框
        save_Path_list = self._get_save_Path_list()
        self.Save_combobox = ttk.Combobox(root, values=save_Path_list,font=("Helvetica", 16))
        self.Save_combobox.config(width=10)
        self.Save_combobox.grid(row=5, column=1, pady=10)
        self.Save_combobox.bind("<<ComboboxSelected>>", self._Save_combobox_on_combobox_select)
        self.Save_combobox.current(2)

    def __init__(self, root):
        self.save_image_path = ""

        self.init_tkinter(root)
        self.Model = AI_Model(self.Model_combobox.get())
        self.save_image_path = self.Save_combobox.get()
    
    def Save_image(self, event=None):
        label = self.E_Correction.get()

        folder = os.path.join("images" , self.save_image_path)
        folder = os.path.join(folder , label)

        randomcode = self._generate_random_code()
        
        filename = "Image_"+  label + "_" + randomcode + ".png"
        Image_path = os.path.join(folder, filename)

        if(not os.path.exists(Image_path)): #資料夾不存在的話 就新增這個資料夾
            os.makedirs(os.path.dirname(Image_path), exist_ok=True)

        print(f"Save Image : {filename} to {Image_path}")
        # 儲存圖片
        self.image.save(Image_path)
        
        # 清空 Canvas 上的內容
        self.canvas.delete("all")
        
        # 重新建立一個黑色背景的圖片
        self.image = Image.new('RGB', (28, 28), 'black')
        self.draw = ImageDraw.Draw(self.image)

        # 更新 Canvas 上的 PhotoImage
        self.canvas_image = tk.PhotoImage(width=self.canvas_size, height=self.canvas_size)
        self.canvas.create_image((self.canvas_size//2, self.canvas_size//2), image=self.canvas_image, anchor=tk.CENTER)
        self.canvas.update_idletasks()

    def Predict(self):

        pred_label = self.Model.Predict(self.image)

        # text = f"{pred_label}"
        text= self._labelmapping_to_str(pred_label)

        pred_label = text
        print(f"Prediction: {pred_label}")
        self.lb_Predict.config(text=f"Prediction: {pred_label}")  # 修改 Label 的文字

        self.E_Correction.delete(0, tk.END)  # 清除輸入框中的內容
        self.E_Correction.insert(0, text)  # 插入新的文字

    def Clear(self):
        # 清空 Canvas 上的內容
        self.canvas.delete("all")
        
        # 重新建立一個黑色背景的圖片
        self.image = Image.new('RGB', (28, 28), 'black')
        self.draw = ImageDraw.Draw(self.image)

        # 更新 Canvas 上的 PhotoImage
        self.canvas_image = tk.PhotoImage(width=self.canvas_size, height=self.canvas_size)
        self.canvas.create_image((self.canvas_size//2, self.canvas_size//2), image=self.canvas_image, anchor=tk.CENTER)
        self.canvas.update_idletasks()

        self.prevPoint = [0,0]
        self.currentPoint = [0,0]

    #region private funtion

    # 畫畫function
    def _paint(self, event):
        x = event.x
        y = event.y
        self.currentPoint = [x // (self.canvas_size // self.image_size), y // (self.canvas_size // self.image_size)]



        if self.prevPoint != [0, 0]:
            # 繪製到tkinter畫布上
            self.canvas.create_line(
                self.prevPoint[0] * (self.canvas_size // self.image_size), self.prevPoint[1] * (self.canvas_size // self.image_size),
                self.currentPoint[0] * (self.canvas_size // self.image_size), self.currentPoint[1] * (self.canvas_size // self.image_size),
                fill="white", width=10,
                capstyle=tk.ROUND
            )

            # 繪製到PIL圖像上
            self.draw.line(
                [self.prevPoint[0], self.prevPoint[1], self.currentPoint[0], self.currentPoint[1]],
                fill="white", width=1  # 使用較小的寬度
            )


        self.prevPoint = self.currentPoint

        if(event.type == "5"): # 鬆開
            self.prevPoint = [0,0]
        
        self.prevPoint = self.currentPoint

    def _paint_reset(self, event):
        self.prevPoint = [0, 0]


    def _labelmapping_to_str(self, label):

        str_label = int(label)
        mapping = self.Model.Label_Mapping

        if mapping != None:
            for key, value in mapping.items(): # {"circle", 1}
                if value == str_label: # 1 == 1
                    return key  # return "circle"
        else:
            raise ValueError("No Label Mapping Exist")


    def _generate_random_code(self, length=15):
        # 定義可用於隨機碼的字元集，包括數字和字母
        characters = string.ascii_letters + string.digits
        
        # 使用隨機選擇來生成隨機碼
        random_code = ''.join(random.choice(characters) for _ in range(length))
        
        return random_code

    def _get_Models_list(self):

        folder = "models"
        filenames = [f for f in os.listdir(folder)]

        model_list = []

        for filename in filenames:
            model_list.append(filename)

        return model_list

    def _Save_combobox_on_combobox_select(self, event):
        
        self.save_image_path = self.Save_combobox.get()

        print(f"Image would keep to Folder : {self.save_image_path}")

    def _Model_combobox_on_combobox_select(self, event):
        # 獲取選擇的模型
        selected_model_name = self.Model_combobox.get()
        self.Model.SwitchModel(selected_model_name)
        print(f"Selected Model: {selected_model_name}")

        self._clear_prediction()
        self.Clear()


    def _get_save_Path_list(self):
        folder = "images"
        foldersnames = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]

        save_Path_list = []

        for filename in foldersnames:
            save_Path_list.append(filename)

        return save_Path_list
    
    def _clear_prediction(self):
        self.lb_Predict.config(text="Not started")

    #endregion


if __name__ == "__main__":
    root = tk.Tk()
    root.resizable(False,False)
    app = Predict_WhatUDraw_App(root)
    root.mainloop()
