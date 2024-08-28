import tkinter as tk
from tkinter import ttk

import os
from PIL import Image, ImageDraw
import random

from Model import AI_Model


class Predict_WhatUDraw_App:

    def init_tkinter(self, root):
        self.root = root
        self.root.title("Drawing App")
        
        self.canvas_size = 280  # 放大顯示區域的尺寸
        self.pixel_size = 10    # 每個像素的顯示大小
        
        # 初始化畫布
        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size, bg='black')
        self.canvas.grid(row=0, column=0, columnspan=4)  # 將畫布放在第一行，橫跨4列
        
        self.canvas.bind("<B1-Motion>", self.paint)
        
        self.image = Image.new('RGB', (28, 28), 'black')
        self.draw = ImageDraw.Draw(self.image)
        
        self.canvas_image = tk.PhotoImage(width=self.canvas_size, height=self.canvas_size)
        self.canvas.create_image((self.canvas_size//2, self.canvas_size//2), image=self.canvas_image, anchor=tk.CENTER)
        
        # 模型選擇下拉框
        models_list = self._get_Models_list()
        self.Model_combobox = ttk.Combobox(root, values=models_list,font=("Helvetica", 16))
        self.Model_combobox.config(width=30)
        self.Model_combobox.grid(row=1, column=0, columnspan=2,padx=30, pady=10)
        self.Model_combobox.bind("<<ComboboxSelected>>", self.Model_combobox_on_combobox_select)
        self.Model_combobox.current(0)

        # 開始預測按鈕
        self.Predict_button = tk.Button(self.root, text="開始預測", command=self.Predict,font=("Helvetica", 16))
        self.Predict_button.grid(row=2, column=0, padx=10)

        # 清除畫板按鈕
        self.Clear_button = tk.Button(self.root, text="清除畫板", command=self.Clear,font=("Helvetica", 16))
        self.Clear_button.grid(row=2, column=1, padx=10)

        # 預測結果無用標籤
        self.lb_label1 = tk.Label(root, text="預測: ", font=("Helvetica", 16))
        self.lb_label1.grid(row=3, column=0, columnspan=4, padx=30,pady=10, sticky='w')

        # 預測結果標籤
        self.lb_Predict = tk.Label(root, text="未開始", font=("Helvetica", 16))
        self.lb_Predict.config(width=20)
        self.lb_Predict.grid(row=3, column=1, columnspan=4, pady=5, sticky='w')

        # 訂正輸入標籤
        self.lb_Correction = tk.Label(root, text="修正: ", font=("Helvetica", 16))
        self.lb_Correction.grid(row=4, column=0, columnspan=4, padx=30,pady=10, sticky='w')
        # 訂正輸入框
        self.E_Correction = tk.Entry(root,font=("Helvetica", 16))
        self.E_Correction.config(width=20)
        self.E_Correction.grid(row=4, column=1, columnspan=4, pady=5, sticky='w')

        # 保存圖片按鈕
        self.Save_button = tk.Button(self.root, text="保存圖片", command=self.save_image ,font=("Helvetica", 16))
        self.Save_button.grid(row=5, column=0, padx=5)

        # 保存選擇下拉框
        self.Save_combobox = ttk.Combobox(root, values=["train", "test"],font=("Helvetica", 16))
        self.Save_combobox.config(width=10)
        self.Save_combobox.grid(row=5, column=1, pady=10)
        self.Save_combobox.bind("<<ComboboxSelected>>", self.Save_combobox_on_combobox_select)
        self.Save_combobox.current(0)

        

    def __init__(self, root):
        self.save_image_path = ""

        self.init_tkinter(root)
        self.Model = AI_Model()
        self.save_image_path = self.Save_combobox.get()

        
        



    def paint(self, event):
        # 計算畫筆在實際圖像中的位置
        x = event.x // self.pixel_size
        y = event.y // self.pixel_size
        
        if 0 <= x < 28 and 0 <= y < 28:
            # 在 Canvas 上畫白色圓點
            self.canvas.create_oval(event.x - self.pixel_size//2, event.y - self.pixel_size//2,
                                    event.x + self.pixel_size//2, event.y + self.pixel_size//2,
                                    fill='white', outline='white')
            # 在 Image 上畫白色圓點
            self.draw.ellipse((x-0.5, y-0.5, x+0.5, y+0.5), fill='white', outline='white')

    def generate_random_code(self, length=10):
        # 建立一個空的字串來儲存隨機碼
        random_code = ''
        
        # 依照指定長度生成隨機數字
        for _ in range(length):
            # 產生一個介於0到9之間的隨機數字
            digit = random.randint(0, 9)
            # 將數字轉成字串並添加到隨機碼中
            random_code += str(digit)
        
        return random_code

    def save_image(self):
        label = self.E_Correction.get()

        folder = os.path.join("images" , self.save_image_path)
        folder = os.path.join(folder , label)

        randomcode = self.generate_random_code()
        
        filename = "Image_"+  label + "_" + randomcode + ".png"
        Image_path = os.path.join(folder, filename)

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

        pred_label = self.Model.predict(self.image)

        text = f"{pred_label}"
        print(f"預測: {pred_label}")
        self.lb_Predict.config(text=f"預測: {pred_label}")  # 修改 Label 的文字

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
    
    
    def Save_combobox_on_combobox_select(self, event):
        
        self.save_image_path = self.Save_combobox.get()

        print(f"圖片 將保存於 : {self.save_image_path} 資料夾")


    def Model_combobox_on_combobox_select(self, event):
        # 獲取選擇的模型
        selected_model_text = self.Model_combobox.get()

        model_folder = "Models"
        selected_model_path = os.path.join(model_folder, selected_model_text)

        self.Model.SwitchModel(selected_model_path)

        print(f"選擇的模型是: {selected_model_path}")



    def _get_Models_list(self):

        folder = "models"
        filenames = [f for f in os.listdir(folder)]

        model_list = []

        for filename in filenames:
            model_list.append(filename)

        return model_list




if __name__ == "__main__":
    root = tk.Tk()
    app = Predict_WhatUDraw_App(root)
    root.mainloop()
