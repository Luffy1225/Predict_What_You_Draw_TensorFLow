import tkinter as tk
import os
from PIL import Image, ImageDraw
import random

class DrawingApp:
    def __init__(self, root):
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

        save_button = tk.Button(self.root, text="Save", command=self.save_image)
        save_button.pack()

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

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
