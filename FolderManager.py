import os


PICTURE_EXT_LIST = ['.jpg', '.png']
PYTHON_EXT_LIST = ['py']





class FolderManager:
    def __init__(self, imagefolder="images", valid_ext = None):
        self.Imagefolder = imagefolder
        self.Valid_ext = valid_ext

    def Count_File(self, valid_ext=None):
        if valid_ext is None:
            valid_ext = self.Valid_ext

        if not os.path.exists(self.Imagefolder):
            print(f"資料夾 '{self.Imagefolder}' 不存在。")
            return

        total_images = 0
        total_size = 0

        # 計算 Imagefolder 本身的照片
        folder_image_count, folder_size = self._count_files_in_folder(self.Imagefolder, valid_ext)
        print(f"{self.Imagefolder} 底下有 {folder_image_count} 個檔案, 總大小: {self._format_size(folder_size)}")

        total_images += folder_image_count
        total_size += folder_size

        # 計算子資料夾中的圖片
        subfolder_names = [d for d in os.listdir(self.Imagefolder) if os.path.isdir(os.path.join(self.Imagefolder, d)) and d != 'ignore']

        print(f"{self.Imagefolder} 底下有 {len(subfolder_names)} 個資料夾 :")
        for subfolder in subfolder_names:
            print(f"資料夾名稱: {subfolder}")
            subfolder_path = os.path.join(self.Imagefolder, subfolder)
            label_folders = [d for d in os.listdir(subfolder_path) if os.path.isdir(os.path.join(subfolder_path, d))]
            subfolder_image_count = 0
            subfolder_size = 0

            for label_folder in label_folders:
                label_folder_path = os.path.join(subfolder_path, label_folder)
                image_count, folder_size = self._count_files_in_folder(label_folder_path, valid_ext)
                
                subfolder_image_count += image_count
                subfolder_size += folder_size
                print(f"  {label_folder}: {image_count} 個檔案, 大小: {self._format_size(folder_size)}")

            print(f"{subfolder} 總共有 {subfolder_image_count} 個檔案, 總大小: {self._format_size(subfolder_size)}\n")
            total_images += subfolder_image_count
            total_size += subfolder_size

        print(f"{self.Imagefolder} ({self._est_tostring(valid_ext)}) 總共有 {total_images} 個檔案, 總大小: {self._format_size(total_size)}")

    def _count_files_in_folder(self, folder_path, valid_ext):
        """
        計算資料夾中的圖片數量和總大小，根據 valid_ext 過濾檔案
        """
        if not os.path.exists(folder_path):
            return 0, 0

        if valid_ext:
            valid_ext = [ext.lower() for ext in valid_ext]
        else:
            valid_ext = []

        image_files = [f for f in os.listdir(folder_path) 
                       if os.path.isfile(os.path.join(folder_path, f)) and 
                          (not valid_ext or os.path.splitext(f)[1].lower() in valid_ext)]

        image_count = len(image_files)
        total_size = sum(os.path.getsize(os.path.join(folder_path, f)) for f in image_files)
        
        return image_count, total_size

    @staticmethod
    def _format_size(size):
        # 格式化檔案大小 (Bytes -> KB, MB, GB)
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.2f} {unit}"
            size /= 1024.0
        return f"{size:.2f} TB"

    @staticmethod
    def _est_tostring(valid_ext): # 把 副檔名 列表 轉換成 字串
        if(valid_ext is not None):
            return " ".join(valid_ext)
        else:
            return "沒有附檔名限制"



if __name__ == "__main__":
    manager = FolderManager("images")
    manager.Count_File()  # 輸入您需要的副檔名
