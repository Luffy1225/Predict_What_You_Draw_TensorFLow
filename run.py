import tkinter as tk
from tkinter import ttk

import os
from PIL import Image, ImageDraw
import random

from Model import MNIST_Model


class Predict_WhatUDraw_App:

    def init_tkinter(self, root):
        self.root = root
        self.root.title("Drawing App")
        
        self.canvas_size = 280  # 放大顯示區域的尺寸
        self.pixel_size = 10    # 每個像素的顯示大小
        
        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size, bg='black')
        self.canvas.pack()
        
        self.canvas.bind("<B1-Motion>", self.paint)
        
        self.image = Image.new('RGB', (28, 28), 'black')
        self.draw = ImageDraw.Draw(self.image)
        
        self.canvas_image = tk.PhotoImage(width=self.canvas_size, height=self.canvas_size)
        self.canvas.create_image((self.canvas_size//2, self.canvas_size//2), image=self.canvas_image, anchor=tk.CENTER)

        # self.save_button = tk.Button(self.root, text="Save", command=self.save_image)
        # self.save_button.pack()


        models_list = self._get_Models_list()

        self.Model_combobox = ttk.Combobox(root, values=models_list)
        self.Model_combobox.pack(pady=10)
        self.Model_combobox.bind("<<ComboboxSelected>>", self.on_combobox_select)
        self.Model_combobox.current(0)

        self.Test_button = tk.Button(self.root, text="開始預測", command=self.Predict)
        self.Test_button.pack(side=tk.LEFT)

        self.Clear_button = tk.Button(self.root, text="清除畫板", command=self.Predict)
        self.Clear_button.pack(side=tk.LEFT)



        self.lb_Predict = tk.Label(root, text="預測: ")
        self.lb_Predict.pack(pady=20, fill='x')


        

    def __init__(self, root):
        self.init_tkinter(root)
        self.Model = MNIST_Model()


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
        folder = "image"
        randomcode = self.generate_random_code()
        filename = "drawing_" + randomcode + ".png"
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

        print(f"預測: {pred_label}")
        self.lb_Predict.config(text=f"預測: {pred_label}")  # 修改 Label 的文字
        self.lb_Predict.pack(pady=20)  # 使用 pack() 方法顯示在視窗中

    def on_combobox_select(self, event):
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
