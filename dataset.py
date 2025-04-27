from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from config import resize_x, resize_y
from torchvision import transforms

# Use a 3-channel normalization for RGB images
default_transform = transforms.Compose([
    transforms.Resize((resize_x, resize_y)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class TrafficSignDataset(Dataset):
    def __init__(self, root_dir='data', transform=None):
        self.images, self.labels = [], []
        data_dir = root_dir
        # Find all class folders
        class_folders = sorted([
            f for f in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, f))
        ])

        # Collect image paths and labels
        for label, class_name in enumerate(class_folders):
            class_path = os.path.join(data_dir, class_name)
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(os.path.join(class_path, img_name))
                    self.labels.append(label)

        # Assign transform (use default if none provided)
        self.transform = transform if transform is not None else default_transform

        # Debug info
        print(f"✅ Dataset initialized with {len(class_folders)} classes.")
        print(f"✅ Total images loaded: {len(self.images)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label


def unicornLoader(dataset=None, shuffle=True):
    from config import batchsize
    ds = dataset if dataset is not None else TrafficSignDataset()
    return DataLoader(ds, batch_size=batchsize, shuffle=shuffle)
