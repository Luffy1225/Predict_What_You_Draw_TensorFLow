import os
import string
import shutil

import tensorflow as tf
from PIL import Image

PICTURE_EXT_LIST = ['.jpg', '.png']
PYTHON_EXT_LIST = ['py']


class FolderManager:
    Default_Imagefolder = "images"

    def __init__(self, imagefolder = "", valid_ext = None):

        if(imagefolder == ""):
            self.Imagefolder = self.Default_Imagefolder
        else:
            self.Imagefolder = imagefolder
       
        self.Valid_ext = valid_ext


    def Count_File(self ,valid_ext=None, Targetperlabel = 5000):
        if valid_ext is None:
            valid_ext = self.Valid_ext

        if not os.path.exists(self.Imagefolder):
            print(f"Folder '{self.Imagefolder}' doesn't exist")
            return

        total_images = 0
        total_size = 0

        # 計算 Imagefolder 本身的照片
        folder_image_count, folder_size = self._count_files_in_folder(self.Imagefolder, valid_ext)
        print(f"Under {self.Imagefolder}, there are {folder_image_count} files, total size: {self._format_size(folder_size)}")

        total_images += folder_image_count
        total_size += folder_size

        # 計算子資料夾中的圖片
        subfolder_names = [d for d in os.listdir(self.Imagefolder) if os.path.isdir(os.path.join(self.Imagefolder, d)) and d != 'ignore']

        print(f"Under {self.Imagefolder}, there are {subfolder_names} folders")
        for subfolder in subfolder_names:
            print(f"Folder Name: {subfolder}")
            subfolder_path = os.path.join(self.Imagefolder, subfolder)
            label_folders = [d for d in os.listdir(subfolder_path) if os.path.isdir(os.path.join(subfolder_path, d))]
            subfolder_image_count = 0
            subfolder_size = 0

            TargetCount = len(label_folders)*Targetperlabel
            
            for label_folder in label_folders:
                label_folder_path = os.path.join(subfolder_path, label_folder)
                image_count, folder_size = self._count_files_in_folder(label_folder_path, valid_ext)
                
                subfolder_image_count += image_count
                subfolder_size += folder_size
                print(f"{label_folder}: {image_count} files, size: {self._format_size(folder_size)}")

            print(f"{subfolder} has a total of {subfolder_image_count} files, size: {self._format_size(subfolder_size)}\n")
            total_images += subfolder_image_count
            total_size += subfolder_size

        print(f"{self.Imagefolder} ({self._ext_tostring(valid_ext)}) has a total of {total_images} files, total size: {self._format_size(total_size)}")

        # ANSI escape codes for gold color
        gold = '\033[33;1m'
        reset = '\033[0m'

        rate = total_images / TargetCount
        print(f"{gold}Currently: {total_images} images. {TargetCount - total_images} images remaining to reach the target of {TargetCount} images. Completion rate = {rate:.2f} %{reset}\n\n")

    def _count_files_in_folder(self, folder_path, valid_ext):
        """
        Calculate the number of images and total size in the folder, filtering files based on `valid_ext`.
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
    def _ext_tostring(valid_ext): # Convert the list of file extensions into a string.
        if(valid_ext is not None):
            return " ".join(valid_ext)
        else:
            return "no EXT restriction"

    @classmethod
    def Build_DataSet_Folders(cls):
        base_folder = cls.Default_Imagefolder
        subfolders = ["ignore", "train", "test"]

        # 數字和字母資料夾名稱
        digits = [str(i) for i in range(10)]
        letters = list(string.ascii_uppercase)

        # 創建主資料夾
        os.makedirs(base_folder, exist_ok=True)

        # 在每個子資料夾中創建數字和字母資料夾
        for subfolder in subfolders:
            subfolder_path = os.path.join(base_folder, subfolder)
            os.makedirs(subfolder_path, exist_ok=True)
            
            for digit in digits:
                os.makedirs(os.path.join(subfolder_path, digit), exist_ok=True)
            
            for letter in letters:
                os.makedirs(os.path.join(subfolder_path, letter), exist_ok=True)

        print(f"Folder :'{base_folder}' has been created.")

    @staticmethod
    def Import_image_to_image(frompath, to_path):  # HASN't TEST YET
        try:
            # 確認來源資料夾是否存在
            if not os.path.isdir(frompath):
                print(f"來源資料夾 {frompath} 不存在。")
                return

            # 檢查目標資料夾是否為空字符串
            if not to_path:
                print("目標資料夾名稱不能是空字符串。")
                return

            # 生成目標資料夾名稱
            target_folder = to_path + '_TARGET'

            # 如果目標資料夾不存在，創建與來源資料夾相同的結構
            if not os.path.isdir(target_folder):
                print(f"目標資料夾 {target_folder} 不存在，正在創建...")
                shutil.copytree(frompath, target_folder, dirs_exist_ok=True)
                print(f"資料夾結構 {target_folder} 已建立完成。")
            else:
                # 如果目標資料夾存在，先備份
                backup_path = target_folder + '_BACKUP'
                if not os.path.isdir(backup_path):
                    shutil.copytree(target_folder, backup_path)
                    print(f"備份資料夾 {target_folder} 到 {backup_path} 完成。")
                else:
                    print(f"備份資料夾 {backup_path} 已存在，請手動處理。")

            # 將來源資料夾中的檔案複製到目標資料夾中
            for root, dirs, files in os.walk(frompath):
                for file in files:
                    src_file = os.path.join(root, file)
                    # 生成相對路徑，以保持資料夾結構
                    rel_path = os.path.relpath(src_file, frompath)
                    dest_file = os.path.join(target_folder, rel_path)
                    dest_folder = os.path.dirname(dest_file)

                    # 創建目標資料夾（如果不存在）
                    os.makedirs(dest_folder, exist_ok=True)

                    # 檢查目標檔案是否已存在
                    if os.path.isfile(dest_file):
                        print(f"{file} 已存在於 {dest_folder}。")
                    else:
                        shutil.copy2(src_file, dest_file)
                        print(f"檔案 {file} 從 {src_file} 複製到 {dest_file} 完成。")

        except Exception as e:
            print(f"導入圖像時發生錯誤: {e}")

    @staticmethod
    def Download_MNIST_DataSet(num_per_label=2000, output_dir="./mnist_images"):
        # download MNIST dataset
        (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

        # 建立基礎資料夾結構
        train_dir = os.path.join(output_dir, "train")
        test_dir = os.path.join(output_dir, "test")
        ignore_dir = os.path.join(output_dir, "ignore")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        os.makedirs(ignore_dir, exist_ok=True)

        # 建立每個標籤的子資料夾
        for label in range(10):
            os.makedirs(os.path.join(train_dir, str(label)), exist_ok=True)
            os.makedirs(os.path.join(test_dir, str(label)), exist_ok=True)
            os.makedirs(os.path.join(ignore_dir, str(label)), exist_ok=True)

        # 記錄每個標籤的儲存數量
        label_count = {i: 0 for i in range(10)}

        total = 0
        for i, (image, label) in enumerate(zip(x_train, y_train)):
            # 如果該標籤已達到 num_per_label，跳過
            if label_count[label] >= num_per_label:
                continue

            # 儲存圖片至 train 資料夾
            label_dir = os.path.join(train_dir, str(label))
            img = Image.fromarray(image)  # 轉換為 PIL 圖片
            img.save(os.path.join(label_dir, f"{label_count[label]}.jpg"))

            # 更新該標籤的儲存數量
            label_count[label] += 1
            total += 1

            # 如果所有標籤都達到 num_per_label，結束
            if all(count >= num_per_label for count in label_count.values()):
                break

        print(f"Save complete! Number of images stored per label: {num_per_label}, total: {total}")







if __name__ == "__main__":
    while True:
        print("\n\n")
        print("Choose an operation:")
        print("1. Create folder structure")
        print("2. Count image files")
        print("3. Import images")
        print("4. Download MNIST training set")
        print("Other. Exit")
        print("Press Enter to default to counting image files")

        choice = input("Enter your choice: ")
        print("\n")


        if choice == '':
            # Default choice: Count image files
            folder_path = input("Please enter the path to the image folder (default is 'images'): ")
            if not folder_path:
                folder_path = "images"
            manager = FolderManager(imagefolder=folder_path, valid_ext=PICTURE_EXT_LIST)
            manager.Count_File()
        
        elif choice == '1':
            # Build the DataSet Folder
            FolderManager.Build_DataSet_Folders()
            print("The folder structure has been created.")
        
        elif choice == '2':
            # Count Image
            folder_path = input("Please enter the path to the image folder (default is 'images'): ")
            if not folder_path:
                folder_path = "images"
            manager = FolderManager(imagefolder=folder_path, valid_ext=PICTURE_EXT_LIST)
            manager.Count_File()
        
        elif choice == '3':
            # Import Image
            from_path = input("Please enter the 'SOURCE' folder path: ")
            to_path = input("Please enter the 'DESTINATION'folder path: ")
            FolderManager.Import_image_to_image(from_path, to_path)

        elif choice == '4':
           
           print("Please enter how many images are needed for each label. (Recommand: 2000)")
           count = int(input("Count:"))

           if(count > 2000):
               print("Too big.") 
           FolderManager.Download_MNIST_DataSet(count, "./mnist_images")

        
        else:
            print("Exit...")
            break
        