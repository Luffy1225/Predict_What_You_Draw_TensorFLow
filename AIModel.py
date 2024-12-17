#region Import

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import json
from PIL import Image
import cv2
from tensorflow.keras.datasets import mnist # type: ignore
#from tensorflow.keras.datasets import emnist # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense # type: ignore

#endregion


class AI_Model:

    def __init__(self, _model_name = ""):

        self.default_model_name = self.Get_Models_List()[0]
        self.target_dataset = "custom" # using my own dataset as Default

        if(_model_name == ""): # If the model name is not specified, use the first entry in the model folder.
            self.Model_Name = self.default_model_name
            self.Model = self.Load_model(self.Model_Name)
        else:
            self.Model_Name = _model_name
            self.Model = self.Load_model(self.Model_Name)

        self.History = None  # Save  parameters and results during the model training process

        self._train_flag = False
        self._Epochs = -1

    #region Public function

    def Load_model(self,model_filename = "" ): # load model by the filename, Load "models/model_name/model_name.keras"
        
        model_valid =  self._select_model_Valid(model_filename)

        if(not model_valid):
            model_filename = self.default_model_name

        self.Model_Name = model_filename.split('.')[0]  # 使用 split 分割字串，取出第一部分

        model_folder = 'models'
        model_path = os.path.join(model_folder, model_filename)
        model_path = os.path.join(model_path, model_filename + ".keras") # Ex: models/model_name/model_name.keras


        self.Model = load_model(model_path)
        self.Label_Mapping = self._load_label_file()
        
        if self.Model is not None:
            print(f"Model: {model_path} load SUCCESSFULLY")
        else:
            print(f"Model: {model_path} load FAIL")
        return self.Model

    def Predict_imageS(self, image_folder): ## Predict multiple Image  的base function
        # get all the image with .png
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
            print("There are no valid images for prediction.")
            return valid_filenames, [], []
        
        images = np.array(images)
        predictions = self.Model.predict(images)
        predicted_labels = np.argmax(predictions, axis=1)
        
        return valid_filenames, images, predicted_labels
    
    def Predict_image_path(self, image_path):
        img = Image.open(image_path)

        predicted_label = self.Predict(img)

        return predicted_label

    def Predict(self, image): # Predict the Single image  的base function

        if(self.Model is not None):

            img = self._load_and_preprocess_image(image) 

            img = np.expand_dims(img, axis=0)  # 增加一維以匹配模型輸入
            prediction = self.Model.predict(img)
            predicted_label = np.argmax(prediction, axis=1)

            predicted_label = predicted_label[0]
            return predicted_label
        else:
            print("Model is None, Please Select Model or New a Model.")

    def SwitchModel(self, model_name): # Switch the model by "Mode name" EX: "model_name_Epoch_100"
        self.Model = self.Load_model(model_name)

    def EvaluatePerformance(self):
        if self.Model is None:
            print("The model has not been loaded yet. Please load the model first.")
            return
        
        (train_images, train_labels), (test_images, test_labels) = self._load_dataset()

        # 數據預處理
        train_images = train_images.astype('float32') / 255
        test_images = test_images.astype('float32') / 255
        train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
        test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
        train_labels = to_categorical(train_labels)
        test_labels = to_categorical(test_labels)

        test_loss, test_acc = self.Model.evaluate(test_images, test_labels)

        print(f'Test set accuracy: {test_acc:.4f}')

        if self._train_flag:
            self._visualizePerformance()
        else:
            print("The model has not been trained, so the training process cannot be displayed.")

    def Train_Model(self, Save_Model = False , epochs = 5): # 
        
        (train_images, train_labels), (test_images, test_labels) = self._load_dataset()  # 預計接口

        # preprocess
        train_images = train_images.astype('float32') / 255
        test_images = test_images.astype('float32') / 255
        train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
        test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
        train_labels = to_categorical(train_labels)
        test_labels = to_categorical(test_labels) 


        train_label_amount = len(train_labels[0])
        print(f"Train has {train_label_amount} labels")
        test_label_amount = len(test_labels[0])
        print(f"Test has {test_label_amount} labels")



        # Model
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(train_label_amount, activation='softmax')
        ])

        model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

        # training
        self._Epochs = epochs

        self.History = model.fit(train_images, train_labels, epochs=self._Epochs, batch_size=64, validation_split=0.2)

        if Save_Model:
            self.Save_Model(model, self._Epochs)

        self.Model = model

        self._train_flag = True
        return model

    def Save_Model(self, _model_name = None):

        if self.Model is not None:

            models_path = "models"  # Root folder 

            _model_name = f'{_model_name}_Epoch_{ self._Epochs}' # model name EX: "model_name_Epoch_100"
            self.Model_Name = _model_name # model name

            models_path = os.path.join(models_path, _model_name) # models/model_name_Epoch_100

            #mkdir models folder
            if not os.path.exists(models_path):
                os.mkdir(models_path)
            
            # save model
            model_full_name = _model_name + ".keras"
            model_full_name = os.path.join(models_path, model_full_name) # models\\testing_Epoch_5\\testing_Epoch_5.keras

            self.Model.save(model_full_name)
            print(f"Save Model {_model_name} to: {model_full_name}")

            # save label file
            label_full_name = _model_name + ".json"
            label_full_name = os.path.join(models_path, label_full_name) # models\\testing_Epoch_5\\testing_Epoch_5.json
            self._save_label_file(label_full_name)
            print(f"Save Label File to: {label_full_name}")


        else:
            print("Model is None.")

    def New_Model(self, name):
        self.Model_Name = name

    def Print_Model_List(self): # list out all the models in "models" folder
        model_list = self.Get_Models_List()

        for index in range(len(model_list)):
            print(f"{index+1}. Model : {model_list[index]}")
    
    def Get_Models_List(self): # Return Valid Model list (names, not the real model)
        folder = "models"
        all_items = os.listdir(folder)

        all_models = []

        for model_name in all_items:
            item_path = os.path.join(folder, model_name)

            # If the model_name is Folder 
            if os.path.isdir(item_path):
                item_path = os.path.join(item_path, model_name)

                ext = ".json" 
                print(item_path + ext)
                if (os.path.exists(item_path + ext)): # Does the Mapping file exist?
                    all_models.append(model_name)

        return all_models
    
    #endregion

    #region private function

    
    def _load_dataset(self): ## mock (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

        if (self.target_dataset == "mnist"):
            (train_images, train_labels), (test_images, test_labels) = self._load_MNIST()
            self._set_mnist_label_mapping()
            
        elif (self.target_dataset == "Emnist"):
            (train_images, train_labels), (test_images, test_labels) = self._load_EMNIST()
            self._set_Emnist_label_mapping()
        elif(self.target_dataset == "custom"): # Using your own Data_set
            (train_images, train_labels) = self._get_train_image_and_label()
            (test_images, test_labels) = self._get_test_image_and_label()
        else:
            raise ValueError(f"Didn't select the target dataset")

        return (train_images, train_labels) ,(test_images, test_labels)

    def _load_EMNIST(self):# NOT YETTTTT
        (train_images, train_labels), (test_images, test_labels) = None # NOT YETTTTT
        return (train_images, train_labels) ,(test_images, test_labels)

    def _load_MNIST(self):
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        return (train_images, train_labels) ,(test_images, test_labels)

    def _load_label_mapping(self):
        json_path = os.path.join("models", self.Model_Name, self.Model_Name + ".json")
        with open(json_path, 'r') as f:
            label_mapping = json.load(f)
        return label_mapping

    def _get_train_image_and_label(self):
        path = os.path.join("images", "train")
        image,label = self._load_images_and_labels(path)

        return (image, label)

    def _get_test_image_and_label(self):
        path = os.path.join("images", "test")
        image,label = self._load_images_and_labels(path)

        return (image, label)

    def _load_images_and_labels(self,folder): # get the images and labels 

        # make sure the folder existed
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Folder '{folder}' NOT FOUND")
        
        # new a list to keep those images and label
        image_list = []
        label_list = []

        # 遍歷資料夾中的所有類別資料夾
        self.Label_Mapping = {}

        # class_trainlabel : int   , class_label : string of the label
        for class_trainlabel, class_label in enumerate(os.listdir(folder)):
            class_folder = os.path.join(folder, class_label)
            
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
                        label_list.append(class_trainlabel)

            self.Label_Mapping[class_label] = class_trainlabel
                    

        print(f"{folder} has {len(image_list)} images")

        if (len(image_list) == 0):
            raise FileNotFoundError(f"Folder '{folder}' has no Valid image")


        # 將圖像和標籤列表轉換為 NumPy 陣列
        images = np.array(image_list)
        labels = np.array(label_list)

        return (images, labels)

    def _visualizePerformance(self):  
        _ , (test_images, test_labels) = self._load_dataset()  # 預計接口

        # 視覺化訓練過程
        if(self.History is not None):   

            # 新增字體
            _font_name = 'taipei_sans_tc_beta.ttf'
            font_path = os.path.join("font",_font_name)

            if( not os.path.exists(font_path)):

                print(f"Can't found font {font_path}")
                return
            
            matplotlib.font_manager.fontManager.addfont(font_path)
            matplotlib.rc('font', family='Taipei Sans TC Beta')

            plt.figure(figsize=(12, 4))

            epochs = list(range(1, len(self.history.history['accuracy']) + 1)) # start form 1

            plt.subplot(1, 2, 1)
            plt.plot(epochs ,self.History.history['accuracy'], label='Training Accuracy')
            plt.plot(epochs , self.History.history['val_accuracy'], label='Validation Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # 設置 x 軸顯示整數刻度



            plt.subplot(1, 2, 2)
            plt.plot(epochs ,self.History.history['loss'], label='Training Loss')
            plt.plot(epochs ,self.History.history['val_loss'], label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # 設置 x 軸顯示整數刻度


            plt.show()

        else:
            print("History is NONE, Please Train the Model")

    def _load_and_preprocess_image(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)
    
        # 檢查 img 是否為 numpy 陣列
        if not isinstance(img, np.ndarray):
            print(f"Invalid image data type: {type(img)}")
            return None
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 灰階
    
        img = cv2.resize(img, (28, 28))  # 調整大小為 28x28
        img = img.astype('float32') / 255  # 正規化
        img = np.expand_dims(img, axis=-1)  # 增加一維以匹配模型輸入
        return img

    def _select_model_Valid(self, model_name):
        model_list = self.Get_Models_List()

        valid = False
        for model in model_list:
            if(model_name == model):
                valid = True
                break
        
        if(valid):
            print(f"Find Model: {model_name}")
        else:
            print(f"Couldn't Find the Model: {model_name}")

        return valid
    
    def _save_label_file(self, label_full_name): # write the label mapping to the json file
        json_path = label_full_name

        with open(json_path, 'w') as f:
            json.dump(self.Label_Mapping, f)
    
    def _load_label_file(self): # load the label mapping from the json file
        json_path = os.path.join("models", self.Model_Name, self.Model_Name + ".json")
        
        if not os.path.exists(json_path):
            return None
        
        try:
            with open(json_path, 'r') as f:
                _label_Mapping = json.load(f)

        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading the label file: {e}")
            return None
        
        # 如果成功讀取，返回 Label_Mapping
        return _label_Mapping


    def _set_mnist_label_mapping(self):
        mapping = {}
        for i in range(10):
            mapping[str(i)] = i
        
        self.Label_Mapping = mapping

    def _set_Emnist_label_mapping(self):
        mapping = {}
        for i in range(10):
            mapping[str(i)] = i
        for i in range(10, 36):
            mapping[chr(i + 55)] = i
        self.Label_Mapping = mapping

    #endregion



if __name__ == '__main__':

    Model = None
    Model = AI_Model()

    while(True):
        print("\n\n")
        print("1. Create a New Model")
        print("2. Load an Existing Model")
        print("3. Start Training")
        print("4. Test Performance")
        print("5. Save the Model")
        print("Other: EXIT")

        input_str = input("Enter a function (1~5): ")



        if(input_str == "1"):
            Model = AI_Model()
            print("\n\nA new model has been created.\n\n")
        elif(input_str == "2"):
            Model.Print_Model_List()
            select_model = input("Enter the 'name' of the model to load: ")
            Model.Load_model(select_model)
        elif(input_str == "3"):
            print("which dataset do you want to use?")
            user_input = input("1: custom, 2: mnist, 3: Emnist: (type 1~3) ")

            options = {
                "1": "custom",
                "2": "mnist",
                "3": "Emnist"
            }

            result = options.get(user_input, "null")
            Model.target_dataset = result

            in_epoch = int(input("Enter the number of training epochs: "))
            Model.Train_Model(epochs=in_epoch)
            Model.EvaluatePerformance()

        elif(input_str == "4"):
            Model.EvaluatePerformance()
        elif(input_str == "5"):
            save_model_name = input("Enter the name for the new model: ")
            Model.Save_Model(save_model_name)
        else:
            break





