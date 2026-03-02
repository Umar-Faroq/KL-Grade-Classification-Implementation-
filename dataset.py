import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(split: str, img_size: int = 224):
    """Return transforms for train / val / test split."""
    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:  # val / test
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


class KLGradeDataset(Dataset):
    """Knee X-ray KL-grade dataset loaded from a CSV split file.

    Expected CSV columns (among others):
        image_path  – absolute path to the PNG image
        xrkl        – integer KL grade label (0–4)
    """

    def __init__(self, csv_path: str, transform=None):
        df = pd.read_csv(csv_path)
        # Drop rows where xrkl is NaN (unannotated images)
        df = df.dropna(subset=["xrkl"])
        self.df = df[["image_path", "xrkl"]].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        label = int(row["xrkl"])
        if self.transform:
            img = self.transform(img)
        return img, label
