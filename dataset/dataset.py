from torchvision import datasets
import torch
from pathlib import Path


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

DATA_DIR = Path("../data")
SPLIT_DIR = DATA_DIR / "split_6000"

SPLIT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42


# --------------------------------------------------
# LOAD ORIGINAL CIFAR-10
# --------------------------------------------------

full_train = datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=False 
)

full_test = datasets.CIFAR10(
    root=DATA_DIR,
    train=False,
    download=False
)

print("Original dataset:")
print("Train:", len(full_train))
print("Test:", len(full_test))


classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


generator = torch.Generator().manual_seed(SEED)


# --------------------------------------------------
# BALANCED SAMPLING FUNCTION
# --------------------------------------------------

def balanced_indices(dataset, samples_per_class):
    """Select an equal number of samples from every class."""

    targets = torch.tensor(dataset.targets)

    selected = []

    for class_id in range(10):

        # Get all indices belonging to this class
        class_indices = torch.where(
            targets == class_id
        )[0]

        # Shuffle the class indices
        shuffled = class_indices[
            torch.randperm(
                len(class_indices),
                generator=generator
            )
        ]

        # Select required number
        selected.extend(
            shuffled[:samples_per_class].tolist()
        )

    return selected


# --------------------------------------------------
# TRAIN + VALIDATION
# --------------------------------------------------

# 540/class = 5400 total
# 480/class -> train
# 60/class  -> validation

train_val_indices = balanced_indices(
    full_train,
    samples_per_class=540
)


train_indices = []
val_indices = []


for class_id in range(10):

    # Get the selected indices for this class
    class_indices = [
        i for i in train_val_indices
        if full_train.targets[i] == class_id
    ]

    # 480 train images/class
    train_indices.extend(
        class_indices[:480]
    )

    # 60 validation images/class
    val_indices.extend(
        class_indices[480:]
    )


# --------------------------------------------------
# TEST
# --------------------------------------------------

# 60/class = 600 total

test_indices = balanced_indices(
    full_test,
    samples_per_class=60
)


# --------------------------------------------------
# EXTRACT ACTUAL DATA
# --------------------------------------------------

def extract_data(dataset, indices):

    images = torch.tensor(
        dataset.data[indices],
        dtype=torch.uint8
    )

    labels = torch.tensor(
        [dataset.targets[i] for i in indices],
        dtype=torch.long
    )

    return images, labels


train_images, train_labels = extract_data(
    full_train,
    train_indices
)

val_images, val_labels = extract_data(
    full_train,
    val_indices
)

test_images, test_labels = extract_data(
    full_test,
    test_indices
)


# --------------------------------------------------
# SAVE SPLITS
# --------------------------------------------------

torch.save(
    {
        "images": train_images,
        "labels": train_labels,
        "classes": classes
    },
    SPLIT_DIR / "train.pt"
)

torch.save(
    {
        "images": val_images,
        "labels": val_labels,
        "classes": classes
    },
    SPLIT_DIR / "val.pt"
)

torch.save(
    {
        "images": test_images,
        "labels": test_labels,
        "classes": classes
    },
    SPLIT_DIR / "test.pt"
)


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print("\nSaved dataset:")
print(f"Train:      {len(train_labels)} images")
print(f"Validation: {len(val_labels)} images")
print(f"Test:       {len(test_labels)} images")

print(f"Total:      {len(train_labels) + len(val_labels) + len(test_labels)} images")

print("\nFiles:")
print(SPLIT_DIR / "train.pt")
print(SPLIT_DIR / "val.pt")
print(SPLIT_DIR / "test.pt")