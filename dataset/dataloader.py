import torch
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path


DATA_DIR = Path("data\split_6000")

# CIFAR-10 normalization
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def load_split(filename):
    data = torch.load(DATA_DIR / filename)

    images = data["images"]
    labels = data["labels"]

    # uint8 -> float
    images = images.float() / 255.0

    # (N, H, W, C) -> (N, C, H, W)
    images = images.permute(0, 3, 1, 2)

    # Normalize using CIFAR-10 mean and std
    images = (images - MEAN) / STD

    return images, labels


def get_dataloaders(batch_size=64, seed=42):

    train_images, train_labels = load_split("train.pt")
    val_images, val_labels = load_split("val.pt")
    test_images, test_labels = load_split("test.pt")

    train_dataset = TensorDataset(train_images, train_labels)
    val_dataset = TensorDataset(val_images, val_labels)
    test_dataset = TensorDataset(test_images, test_labels)

    # Fixed generator makes training order reproducible
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader