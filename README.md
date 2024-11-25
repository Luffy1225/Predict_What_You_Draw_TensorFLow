# Predict_What_You_Draw

This is a hand-drawn image recognition application based on the Tkinter and PIL libraries. Users can draw digits or letters within the application and make predictions using a pre-trained MNIST model or a custom-trained model.

# Install 

```
git clone https://github.com/Luffy1225/Mnist_Machine_Learning_Practice.git
cd Predict_What_You_Draw_TensorFLow
pip install -r requirements.txt
```


# Run

> [!IMPORTANT]
> Before running, ensure that the `images` folder exists (otherwise, the corresponding training set cannot be found).
> You can extract the provided images_DATE archive and rename it to `images`.

## Start Predict What You Draw

```
python main.py
```

## Start FolderManager
```
python FolderManager.py
```

## Start AI Training
```
python AIModel.py
```

## Feature of Predict What You Draw 

1. Drawing Canvas:
    Users can freely draw digits on the canvas provided by the application.

2. Model Selection:
    Multiple pre-trained models are available for users to choose from and switch between for predictions.

    [!NOTE]
    It is recommended to use the mnist_model_Epoch_100.keras model.

3. Prediction Result Display:
    The application displays the predicted digit below the canvas.

4. Clear Canvas:
    Users can clear all content on the canvas to redraw digits.

5. Save Image:
    Drawn images can be saved locally as part of the training set.


## Structure of `images` Folder

    images/
    ├── ignore/                 # Ignored data, not used for training or testing
    │   ├── label_1/            # Images for label 1
    │   │   ├── 0.jpg           # Image file
    │   │   ├── 1.jpg
    │   │   └── ...             # More images
    │   ├── label_2/            # Images for label 2
    │   │   ├── 0.jpg
    │   │   ├── 1.jpg
    │   │   └── ...
    │   └── label_x/            # Images for other labels
    │       ├── 0.jpg
    │       ├── 1.jpg
    │       └── ...
    ├── test/                   # Test data, used for validating the model
    │   ├── label_1/            # Test images for label 1
    │   │   ├── 0.jpg
    │   │   ├── 1.jpg
    │   │   └── ...
    │   ├── label_2/            # Test images for label 2
    │   │   ├── 0.jpg
    │   │   ├── 1.jpg
    │   │   └── ...
    │   └── label_x/            # Test images for other labels
    │       ├── 0.jpg
    │       ├── 1.jpg
    │       └── ...
    ├── train/                  # Training data, used for training the model
    │   ├── label_1/            # Training images for label 1
    │   │   ├── 0.jpg
    │   │   ├── 1.jpg
    │   │   └── ...
    │   ├── label_2/            # Training images for label 2
    │   │   ├── 0.jpg
    │   │   ├── 1.jpg
    │   │   └── ...
    │   └── label_x/            # Training images for other labels
    │       ├── 0.jpg
    │       ├── 1.jpg
    │       └── ...




- [中文版 說明](README_zh_TW.md)
