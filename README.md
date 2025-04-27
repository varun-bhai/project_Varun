# Traffic Sign Classification 🚦

## Problem Description
Traffic sign classification is an important task for autonomous driving systems. The goal of this project is to build a **Convolutional Neural Network (CNN)** model that classifies traffic sign images into **7 different classes**. Each class represents a particular type of traffic sign (e.g., stop signs, speed limits, warnings, etc.). 

The dataset used consists of images organized into 7 folders (0–6), each folder representing one class.

---

## Input-Output
- **Input**: Traffic sign images (`.jpg`, `.jpeg`, `.png` formats).
- **Output**: A predicted class label (0–6) corresponding to the traffic sign type.

---

## Data Source
The dataset is locally available in the `data/` folder.  
- 7 class folders: `0`, `1`, `2`, `3`, `4`, `5`, `6`.
- Each folder contains images related to that particular class.

---

## Model Architecture
The model used for this project is a **Convolutional Neural Network (CNN)** defined in `model.py`.  
It consists of:

- 4 Convolutional layers:
  - Conv2d (input → 32 filters)
  - Conv2d (32 → 64 filters)
  - Conv2d (64 → 128 filters)
  - Conv2d (128 → 256 filters)
- Each conv layer is followed by **ReLU activation** and **MaxPooling**.
- 1 Fully connected layer with **Dropout (0.25)** for regularization.
- Final output layer with 7 units (one for each class).

**Loss Function**: CrossEntropyLoss  
**Optimizer**: Adam

---

## Configuration
All important hyperparameters are defined in `config.py`:
- Batch size: `32`
- Number of epochs: `5`
- Image size: `64x64`
- Input channels: `3` (RGB images)
- Learning rate: `0.001`
- Number of classes: `7`
- Dataset directory: `data/`

---

## Files and Directories
- `config.py`: All configurable parameters.
- `dataset.py`: Custom PyTorch dataset and data loader.
- `model.py`: CNN model architecture.
- `train.py`: Training loop.
- `predict.py`: Inference functions to predict on images or validate folders.
- `interface.py`: Links all important modules together.
- `data/`: Contains class folders with images.
- `checkpoints/`: Directory to save trained model weights (`final_weights.pth`).

---

## Downloading Dataset
The dataset used here is organized manually into 7 folders and does not require separate downloading.

---

## Training the Model
To train the model, simply run:

```bash
python3 train_test.py
```

- The script will initialize the dataset, create the model, and start training.
- Training will run for 5 epochs.
- Final weights will be saved in the `checkpoints/` folder as `final_weights.pth`.

---

## Using the Predict Function
There are two ways to use predictions:

1. **Single/Multiple Image Paths**  
   Use `cryptic_inf_f(image_paths)` function with a list of image paths to predict their classes.

2. **Validate Data Folder**  
   Use `validate_data_folder('path_to_folder')` to validate a folder containing subfolders of classes (each subfolder should have at least 10 images).

Example:

```python
from predict import cryptic_inf_f, validate_data_folder

images = validate_data_folder("data/")
predictions = cryptic_inf_f(images)
print(predictions)
```

---

## Evaluation Metric
- **Accuracy** is used to measure the model’s performance:  
  > (Number of correct predictions) / (Total predictions)


---