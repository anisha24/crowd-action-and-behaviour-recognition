import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class Crowd11(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_paths = [os.path.join(root_dir, 'Images', img) for img in os.listdir(os.path.join(root_dir, 'Images')) if img.endswith('.jpg')]
        self.annotation_paths = [os.path.join(root_dir, 'Annotations', ann) for ann in os.listdir(os.path.join(root_dir, 'Annotations')) if ann.endswith('.json')]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        ann_path = self.annotation_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

transform = transforms.Compose([
    transforms.Resize((224, 224)), #256
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

crowd_dataset = Crowd11(root_dir='data/crowd11/', transform=transform)