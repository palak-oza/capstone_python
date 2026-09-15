
import torch
import torch.nn as nn
import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# CIFAR-10 CLASS NAMES
# ============================================================

CLASSES = [
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


# ============================================================
# EVALUATE ONE MODEL
# ============================================================

def evaluate_model(model, model_path, test_loader, device):
    """
    Load an already-trained .pth model and evaluate it.

    No training is performed.
    """

    print("\n" + "=" * 60)
    print("LOADING MODEL")
    print("=" * 60)

    print("Model:", model_path)

    # --------------------------------------------------------
    # Load trained weights
    # --------------------------------------------------------

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

   
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_predictions = []
    all_labels = []

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            total_loss += loss.item() * images.size(0)

            total_correct += (
                predictions == labels
            ).sum().item()

            total_samples += labels.size(0)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    test_loss = total_loss / total_samples

    accuracy = total_correct / total_samples

    precision = precision_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(10))
    )

    # --------------------------------------------------------
    # Per-class metrics
    # --------------------------------------------------------

    class_accuracy = {}
    class_precision = {}
    class_recall = {}
    class_f1 = {}

    precision_values = precision_score(
        all_labels,
        all_predictions,
        labels=list(range(10)),
        average=None,
        zero_division=0
    )

    recall_values = recall_score(
        all_labels,
        all_predictions,
        labels=list(range(10)),
        average=None,
        zero_division=0
    )

    f1_values = f1_score(
        all_labels,
        all_predictions,
        labels=list(range(10)),
        average=None,
        zero_division=0
    )

    for i, class_name in enumerate(CLASSES):

        class_total = cm[i].sum()

        if class_total > 0:
            class_accuracy[class_name] = (
                cm[i, i] / class_total
            )
        else:
            class_accuracy[class_name] = 0.0

        class_precision[class_name] = precision_values[i]
        class_recall[class_name] = recall_values[i]
        class_f1[class_name] = f1_values[i]

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\nRESULTS")
    print("-" * 60)

    print(f"Test Loss : {test_loss:.4f}")
    print(f"Accuracy  : {accuracy * 100:.2f}%")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print(f"F1 Score  : {f1 * 100:.2f}%")

    # --------------------------------------------------------
    # Return everything to Jupyter
    # --------------------------------------------------------

    return {
        "loss": test_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "class_accuracy": class_accuracy,
        "class_precision": class_precision,
        "class_recall": class_recall,
        "class_f1": class_f1
    }


# ============================================================
# EVALUATE ALL MODELS
# ============================================================

def evaluate_all_models(
    model_paths,
    model_builders,
    test_loader,
    device
):
    """
    Evaluate all trained models.

    model_paths format:

    {
        "Simple + Adam": {
            "MLP": "...pth",
            "CNN": "...pth"
        },
        ...
    }

    model_builders format:

    {
        "Simple + Adam": {
            "MLP": mlp_builder,
            "CNN": cnn_builder
        },
        ...
    }
    """

    results = {}

    for setting in model_paths:

        print("\n\n")
        print("#" * 70)
        print(setting)
        print("#" * 70)

        results[setting] = {}

        for model_name in ["MLP", "CNN"]:

            model_path = model_paths[setting][model_name]

            print("\n")
            print("-" * 60)
            print(model_name)
            print("-" * 60)

            model = model_builders[setting][model_name]()
            metrics = evaluate_model(
                model=model,
                model_path=model_path,
                test_loader=test_loader,
                device=device
            )

            results[setting][model_name] = metrics

    return results


# ============================================================
# CONVERT RESULTS TO DATAFRAME
# ============================================================

def results_to_dataframe(results):

    import pandas as pd

    rows = []

    for setting, models in results.items():

        for model_name, metrics in models.items():

            rows.append({
                "Setting": setting,
                "Model": model_name,
                "Test Loss": metrics["loss"],
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1": metrics["f1"]
            })

    return pd.DataFrame(rows)

