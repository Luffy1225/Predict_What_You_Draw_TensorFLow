import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import cv2
from tensorflow.keras.datasets import mnist # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense # type: ignore



class AI_Model:

    def __init__(self, default_model_name = "mnist_model_Epoch_60.h5"):
        self.default_model_name = default_model_name
        self.Model_Name = ""
        # self.Model = self.Load_model(default_model_name)
        self.Model = None

        self.History = None  # 保存模型訓練過程中各種參數和結果

        self._train_flag = False
        self._Epochs = -1


    def Load_model(self,model_filename = "" ):
        # model_filename = 'mnist_model_Epoch_60.h5'  # 修改為實際模型文件名
        
        model_valid =  self._select_model_Valid(model_filename)

        if(not model_valid):
            model_filename = self.default_model_name

        self.Model_Name = model_filename.split('.')[0]  # 使用 split 分割字串，取出第一部分

        model_folder = 'models'
        model_path = os.path.join(model_folder, model_filename)

        
        self.Model = load_model(model_path)
        if self.Model is not None:
            print(f"模型: {model_path} 成功載入")
        else:
            print(f"模型: {model_path} 載入失敗")
        return self.Model


    def Predict_imageS(self, image_folder): ## Predict 多張圖片 的base function
        # 獲取資料夾中的所有檔案
        filenames = [f for f in os.listdir(image_folder) if f.endswith('.png')]
        images = []
        valid_filenames = []
        
        for filename in filenames:
            img_path = os.path.join(image_folder, filename)
            img = self._load_and_preprocess_imagePath(img_path)
            if img is not None:  # 只添加有效的圖片
                images.append(img)
                valid_filenames.append(filename)
        
        if len(images) == 0:
            print("沒有有效的圖片進行預測。")
            return valid_filenames, [], []
        
        images = np.array(images)
        predictions = self.Model.predict(images)
        predicted_labels = np.argmax(predictions, axis=1)
        
        return valid_filenames, images, predicted_labels
    
    def Predict_image_path(self, image_path):
        img = Image.open(image_path)

        predicted_label = self.Predict(img)

        return predicted_label

    def Predict(self, image): # Predict 單張圖片 的base function

        if(self.Model is not None):

            # img = PIL Image 
            img = self._load_and_preprocess_image(image) 

            img = np.expand_dims(img, axis=0)  # 增加一維以匹配模型輸入
            prediction = self.Model.predict(img)
            predicted_label = np.argmax(prediction, axis=1)

            predicted_label = predicted_label[0]
            return predicted_label
        else:
            print("Model is None, Please Select Model or New a Model.")

    def SwitchModel(self, model_path):
        self.Model = load_model(model_path)

    def EvaluatePerformance(self, use_self_test_data : bool = False):
        if self.Model is None:
            print("模型尚未載入，請先載入模型")
            return
        
        (train_images, train_labels), (test_images, test_labels) = self._load_train_data()

        # 數據預處理
        # train_images = train_images.astype('float32') / 255
        test_images = test_images.astype('float32') / 255
        # train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
        test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
        # train_labels = to_categorical(train_labels)
        test_labels = to_categorical(test_labels)

        test_loss, test_acc = self.Model.evaluate(test_images, test_labels)

        print(f'測試集準確度: {test_acc:.4f}')

        if self._train_flag:
            self._visulizePerformance()
        else:
            print("模型尚未訓練，無法顯示訓練過程。")

    def Train_Model(self, Save_Model = False , epochs = 5, use_sel_train_data : bool = False): # 訓練模型的base function
        (train_images, train_labels), (test_images, test_labels) = self._load_train_data(use_sel_train_data)  # 預計接口

        # 數據預處理
        train_images = train_images.astype('float32') / 255
        test_images = test_images.astype('float32') / 255
        train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
        test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
        train_labels = to_categorical(train_labels)
        test_labels = to_categorical(test_labels)


        label_amount = len(train_labels[0])


        # 假設這是你的模型定義和訓練代碼
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(label_amount, activation='softmax')
        ])

        model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

        # 訓練模型
        self._Epochs = epochs

        self.History = model.fit(train_images, train_labels, epochs=self._Epochs, batch_size=64, validation_split=0.2)

        if Save_Model:
            self.Save_Model(model, self._Epochs)

        self.Model = model

        self._train_flag = True
        return model

    def Save_Model(self, Model_name = None):

        if self.Model is not None:

            models_path = "models"  # 保存路徑資料夾
            
            # 保存模型
            model_name = self.Model_Name # 模型名稱
            model_full_name = f'{model_name}_Epoch_{ self._Epochs}.h5'
            model_full_name = os.path.join(models_path, model_full_name)

            print(f"Model {model_name}保存到: {model_full_name}")

            self.Model.save(model_full_name)
        else:
            print("Model is None.")

    def New_Model(self, name):
        self.Model_Name = name

    def Print_Model_List(self):
        model_list = self.Get_Models_list()

        for index in range(len(model_list)):
            print(f"{index+1}. Model : {model_list[index]}")

    def Get_Models_list(self):

        folder = "models"
        filenames = [f for f in os.listdir(folder)]

        model_list = []

        for filename in filenames:
            model_list.append(filename)

        return model_list

    #region private function



    # 主要接口
    def _load_train_data(self, use_sel_train_data : bool = False): ## 仿製 (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

        if(use_sel_train_data):
            (train_images, train_labels) = self._get_train_image_and_label()
            (test_images, test_labels) = self._get_test_image_and_label()
        else:
            (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

        return (train_images, train_labels) ,(test_images, test_labels)

    def _get_train_image_and_label(self):
        path = os.path.join("images", "train")
        image,label = self._image_and_label(path)

        return (image, label)

    def _get_test_image_and_label(self):
        path = os.path.join("images", "test")
        image,label = self._image_and_label(path)

        return (image, label)

    def _image_and_label(self,train_folder): # 取得 images 和 label 的 base function
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

        # 將圖像和標籤列表轉換為 NumPy 陣列
        images = np.array(image_list)
        labels = np.array(label_list)

        return (images, labels)


    def _visulizePerformance(self):  
        _ , (test_images, test_labels) = self._load_train_data()  # 預計接口

        # 視覺化訓練過程
        if(self.History is not None):   

            # 新增字體
            _font_name = 'taipei_sans_tc_beta.ttf'
            font_path = os.path.join("font",_font_name)


            if( not os.path.exists(font_path)):

                print(f"找不到 font {font_path}")
                return
            
            matplotlib.font_manager.fontManager.addfont(font_path)
            matplotlib.rc('font', family='Taipei Sans TC Beta')


            plt.figure(figsize=(12, 4))

            plt.subplot(1, 2, 1)
            plt.plot(self.History.history['accuracy'], label='訓練準確度')
            plt.plot(self.History.history['val_accuracy'], label='驗證準確度')
            plt.xlabel('訓練週期')
            plt.ylabel('準確度')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(self.History.history['loss'], label='訓練損失')
            plt.plot(self.History.history['val_loss'], label='驗證損失')
            plt.xlabel('訓練週期')
            plt.ylabel('損失')
            plt.legend()

            plt.show()

        else:
            print("History is NONE, Please Train the Model")

    def _load_and_preprocess_image(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)
    
        # 檢查 img 是否為 numpy 陣列
        if not isinstance(img, np.ndarray):
            print(f"無效的圖片數據類型: {type(img)}")
            return None
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 灰階
    
        img = cv2.resize(img, (28, 28))  # 調整大小為 28x28
        img = img.astype('float32') / 255  # 正規化
        img = np.expand_dims(img, axis=-1)  # 增加一維以匹配模型輸入
        return img

    def _select_model_Valid(self, model_name):
        model_list = self.Get_Models_list()

        valid = False
        for model in model_list:
            if(model_name == model):
                valid = True
                break
        
        if(valid):
            print(f"成功找到 Model: : {model_name}")
        else:
            print(f"未找到 Model: {model_name}")

        return valid
    

    def _download_font(self): # (ABANDOM)不要使用 (請使用_check_and_download_font)
        pass
        # # 確保 font 資料夾存在
        # os.makedirs('font', exist_ok=True)

        # # 下載字型檔案
        # url = "https://drive.google.com/uc?id=1eGAsTN1HBpJAkeVM57_C7ccp7hbgSz3_&export=download"
        # output_file = 'font/taipei_sans_tc_beta.ttf'

        # # 使用 wget 下載檔案
        # try:
        #     subprocess.run(['wget', '-O', output_file, url], check=True)
        #     print(f'File downloaded successfully and saved to {output_file}')
        # except subprocess.CalledProcessError as e:
        #     print(f'Error occurred while downloading the file: {e}')

    def _check_and_download_font(self, check_font_path):# (ABANDOM)
        pass
        # passflag = False

        # if not os.path.exists(check_font_path):
        #     print(f"找不到 font {check_font_path}")
        #     print("嘗試自動下載...")

        #     # 嘗試下載字型檔案
        #     self._download_font()
            
        #     # 再次檢查字型檔案是否存在
        #     if not os.path.exists(check_font_path):
        #         print(f"字型檔案仍然不存在: {check_font_path}")
        #         print("請手動下載字型檔案並將其放置在 'font' 資料夾中。")
        #     else:
        #         print(f"字型檔案下載成功: {check_font_path}")
        #         passflag = True
        # else:
        #     print(f"字型檔案已存在: {check_font_path}")
        #     passflag = True

        # return passflag

    #endregion



if __name__ == '__main__':

    Model = None
    Model = AI_Model()

    while(True):
        print("\n\n")
        print("1. 建立新Model")
        print("2. 載入Model")
        print("3. 開始訓練Model")
        print("4. 測試性能")
        print("5. 保存Model")
        print("其他. 退出")
        input_str = input("輸入功能:(1~5):  ")


        if(input_str == "1"):
            Model = AI_Model()
            print("\n\n已建立新Model\n\n")
        elif(input_str == "2"):
            Model.Print_Model_List()
            select_model = input("請輸入要載入的Model : ")
            Model.Load_model(select_model)
        elif(input_str == "3"):
            mnistOrNOtstr = input("要不要用 MNIST訓練集, 不要 : 使用自己的images訓練集 (y/n)")

            flag = False
            if(mnistOrNOtstr == "y"): flag = False
            else: flag = True

            in_epoch = int(input("輸入訓練週期(Epochs): "))
            Model.Train_Model(epochs = in_epoch, use_sel_train_data=flag)
        elif(input_str == "4"):
            Model.EvaluatePerformance()
        elif(input_str == "5"):
            save_model_name = input("輸入 新Model名稱: ")
            Model.Save_Model()
        else:
            break




