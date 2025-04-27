# Skin Disease Classification Using AI 🩺

## Problem Description
Skin diseases, such as **melanoma**, **eczema**, and **psoriasis**, impact millions worldwide. Early and accurate diagnosis is critical, yet access to dermatologists is often limited. This project develops a **Convolutional Neural Network (CNN)** to classify skin lesion images into **7 disease categories**, enabling faster and automated diagnostics.

The dataset consists of dermatoscopic images organized into 7 folders (0–6), each representing a specific disease class.

---

## Input-Output
- **Input**: High-resolution skin lesion images (`.jpg` or `.png` formats).
- **Output**: A predicted class label (0–6) corresponding to the disease type.

---

## Data Source
The project utilizes the **HAM10000 Dataset** from Kaggle, containing over **10,000 dermatoscopic images** across 7 classes:  
[Skin Cancer MNIST: HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

**Folder Structure**:
```
data/
├── 0/    # Class 0 images
├── 1/    # Class 1 images
│   ⋮
└── 6/    # Class 6 images
```

---

## Model Architecture
Defined in `model.py`, the **MyCustomModel** CNN includes:
- **Four convolutional blocks**:
  - `Conv2d` → ReLU → `MaxPool2d`
  - Filters: 3 → 32 → 64 → 128 → 256
- **Dropout**: p=0.25
- **Fully connected layers**:
  - 256×4×4 → 512 → 7
- **Softmax**: Handled by `CrossEntropyLoss`

---

## Configuration
Hyperparameters are specified in `config.py`:
```python
batchsize      = 32
epochs         = 5
resize_x, y    = 64, 64
input_channels = 3
num_classes    = 7
learning_rate  = 0.001
data_path      = "data/"
loss_fn        = nn.CrossEntropyLoss()
```

---

## Project Structure
```
├── checkpoints/           # Saved model weights
│   └── final_weights.pth
├── data/                  # 7 class folders (0–6)
├── config.py              # Hyperparameters
├── dataset.py             # TrafficSignDataset + unicornLoader
├── model.py               # CNN architecture
├── train.py               # Training loop function
├── predict.py             # validate_data_folder & cryptic_inf_f
├── interface.py           # Connects model/train/predict
├── train_test.py          # End-to-end training & evaluation script
└── README.md              # This file
```

---

## Setup & Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```
2. Create and activate a Python 3 virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install torch torchvision pillow
   ```

---

## Training the Model
Ensure the `data/` folder is organized, then run:
```bash
python3 train_test.py
```
- Trains for **5 epochs**, displaying training loss and validation accuracy per epoch.
- Saves the best model to `checkpoints/final_weights.pth`.

---

## Running Inference
Perform predictions with:
```python
from predict import validate_data_folder, cryptic_inf_f

# Gather 10 sample images per class
imgs = validate_data_folder("data/")

# Predict on the first 5 images
preds = cryptic_inf_f(imgs[:5])
print(preds)  # Example output: [2, 0, 6, 1, 1]
```

---

## Evaluation Metric
**Accuracy** = (Number of correct predictions) / (Total predictions)

---

## Model Code Snippet (`model.py`)
```python
import torch.nn as nn
from config import num_classes

class MyCustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = self.pool(nn.functional.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(nn.functional.relu(self.fc1(x)))
        return self.fc2(x)
```

---

## Contributing
Experiment with hyperparameter tuning, data augmentations, or additional datasets. Contributions are welcome! 🚀