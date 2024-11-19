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

> [!NOTE]
> It is recommended to use the mnist_model_Epoch_100.keras model.

3. Prediction Result Display:
    The application displays the predicted digit below the canvas.

4. Clear Canvas:
    Users can clear all content on the canvas to redraw digits.

5. Save Image:
    Drawn images can be saved locally as part of the training set.

- [中文版 說明](README_zh_TW.md)
