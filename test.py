import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from architecture.mlparc import mlp_arc
from architecture.convnn import cnn_arc
from dataset.dataloader import get_dataloaders


# ==================================================
# SETTINGS
# ==================================================

BATCH_SIZE = 64
SEED = 42

CLASS_NAMES = [
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


# ==================================================
# DEVICE
# ==================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("TEST-ONLY EVALUATION")
print("=" * 60)

print(f"Device: {device}")


# ==================================================
# GET MODEL PATHS
# ==================================================

mlp_path = input(
    "\nEnter path to MLP model (.pth): "
).strip()

cnn_path = input(
    "Enter path to CNN model (.pth): "
).strip()


if not os.path.isfile(mlp_path):
    raise FileNotFoundError(
        f"MLP model not found:\n{mlp_path}"
    )

if not os.path.isfile(cnn_path):
    raise FileNotFoundError(
        f"CNN model not found:\n{cnn_path}"
    )


# ==================================================
# LOAD TEST DATA ONLY
# ==================================================

print("\n" + "=" * 60)
print("LOADING TEST DATA")
print("=" * 60)

_, _, test_loader = get_dataloaders(
    batch_size=BATCH_SIZE,
    seed=SEED
)

print("Test data loaded.")


# ==================================================
# LOAD MODEL
# ==================================================

def load_model(model, model_path):

    model = model.to(device)

    state_dict = torch.load(
        model_path,
        map_location=device
    )

    model.load_state_dict(state_dict)

    model.eval()

    return model


# ==================================================
# EVALUATE MODEL
# ==================================================

def evaluate_model(model, test_loader):

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    correct_examples = {
        class_id: None
        for class_id in range(10)
    }

    wrong_examples = {
        class_id: None
        for class_id in range(10)
    }

    model.eval()

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            predictions = outputs.argmax(
                dim=1
            )

            total_loss += (
                loss.item() * images.size(0)
            )

            total_correct += (
                predictions == labels
            ).sum().item()

            total_samples += labels.size(0)

            # --------------------------------------
            # SAVE CORRECT / WRONG EXAMPLES
            # --------------------------------------

            for i in range(images.size(0)):

                true_label = labels[i].item()
                predicted_label = predictions[i].item()

                # One correct example per class
                if (
                    predicted_label == true_label
                    and correct_examples[true_label] is None
                ):
                    correct_examples[true_label] = (
                        images[i].detach().cpu(),
                        true_label,
                        predicted_label
                    )

                # One wrong example per class
                if (
                    predicted_label != true_label
                    and wrong_examples[true_label] is None
                ):
                    wrong_examples[true_label] = (
                        images[i].detach().cpu(),
                        true_label,
                        predicted_label
                    )

    average_loss = total_loss / total_samples

    accuracy = (
        total_correct / total_samples
    )

    return (
        average_loss,
        accuracy,
        correct_examples,
        wrong_examples
    )


# ==================================================
# VISUALIZE EXAMPLES
# ==================================================

def visualize_examples(
    correct_examples,
    wrong_examples,
    model_name
):

    # ------------------------------------------------
    # CORRECT EXAMPLES
    # ------------------------------------------------

    plt.figure(figsize=(15, 8))

    for class_id in range(10):

        example = correct_examples[class_id]

        if example is None:
            continue

        image, true_label, predicted_label = example

        # Convert CHW -> HWC
        image = image.permute(
            1, 2, 0
        )

        # Undo CIFAR normalization
        mean = torch.tensor(
            [0.4914, 0.4822, 0.4465]
        )

        std = torch.tensor(
            [0.2470, 0.2435, 0.2616]
        )

        image = image * std + mean

        image = torch.clamp(
            image,
            0,
            1
        )

        ax = plt.subplot(
            2,
            5,
            class_id + 1
        )

        ax.imshow(image)

        ax.set_title(
            f"True: {CLASS_NAMES[true_label]}\n"
            f"Pred: {CLASS_NAMES[predicted_label]}"
        )

        ax.axis("off")

    plt.suptitle(
        f"{model_name} - Correct Predictions",
        fontsize=16
    )

    plt.tight_layout()

    plt.show()


    # ------------------------------------------------
    # WRONG EXAMPLES
    # ------------------------------------------------

    plt.figure(figsize=(15, 8))

    for class_id in range(10):

        example = wrong_examples[class_id]

        if example is None:
            continue

        image, true_label, predicted_label = example

        # Convert CHW -> HWC
        image = image.permute(
            1,
            2,
            0
        )

        # Undo CIFAR normalization
        mean = torch.tensor(
            [0.4914, 0.4822, 0.4465]
        )

        std = torch.tensor(
            [0.2470, 0.2435, 0.2616]
        )

        image = image * std + mean

        image = torch.clamp(
            image,
            0,
            1
        )

        ax = plt.subplot(
            2,
            5,
            class_id + 1
        )

        ax.imshow(image)

        ax.set_title(
            f"True: {CLASS_NAMES[true_label]}\n"
            f"Pred: {CLASS_NAMES[predicted_label]}"
        )

        ax.axis("off")

    plt.suptitle(
        f"{model_name} - Wrong Predictions",
        fontsize=16
    )

    plt.tight_layout()

    plt.show()


# ==================================================
# MLP TEST
# ==================================================

print("\n" + "=" * 60)
print("EVALUATING MLP")
print("=" * 60)

mlp_model = mlp_arc()

mlp_model = load_model(
    mlp_model,
    mlp_path
)

(
    mlp_test_loss,
    mlp_test_accuracy,
    mlp_correct,
    mlp_wrong
) = evaluate_model(
    mlp_model,
    test_loader
)

print(
    f"MLP Test Loss: "
    f"{mlp_test_loss:.4f}"
)

print(
    f"MLP Test Accuracy: "
    f"{mlp_test_accuracy * 100:.2f}%"
)


# ==================================================
# CNN TEST
# ==================================================

print("\n" + "=" * 60)
print("EVALUATING CNN")
print("=" * 60)

cnn_model = cnn_arc()

cnn_model = load_model(
    cnn_model,
    cnn_path
)

(
    cnn_test_loss,
    cnn_test_accuracy,
    cnn_correct,
    cnn_wrong
) = evaluate_model(
    cnn_model,
    test_loader
)

print(
    f"CNN Test Loss: "
    f"{cnn_test_loss:.4f}"
)

print(
    f"CNN Test Accuracy: "
    f"{cnn_test_accuracy * 100:.2f}%"
)


# ==================================================
# VISUALIZE
# ==================================================

print("\n" + "=" * 60)
print("VISUALIZING MLP")
print("=" * 60)

visualize_examples(
    mlp_correct,
    mlp_wrong,
    "MLP"
)


print("\n" + "=" * 60)
print("VISUALIZING CNN")
print("=" * 60)

visualize_examples(
    cnn_correct,
    cnn_wrong,
    "CNN"
)


# ==================================================
# FINAL SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(
    f"MLP | "
    f"Accuracy: {mlp_test_accuracy * 100:.2f}% | "
    f"Loss: {mlp_test_loss:.4f}"
)

print(
    f"CNN | "
    f"Accuracy: {cnn_test_accuracy * 100:.2f}% | "
    f"Loss: {cnn_test_loss:.4f}"
)

print("=" * 60)