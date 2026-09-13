
import os
import argparse
import torch

from architecture.mlparc import mlp_arc
from architecture.convnn import cnn_arc

from dataset.dataloader import get_dataloaders
from train import train_model

from metrics import (
    evaluate_model,
    plot_training_curves,
    plot_confusion_matrices,
    save_metrics
)


# ==================================================
# COMMAND-LINE ARGUMENTS
# ==================================================

parser = argparse.ArgumentParser(
    description="Train and evaluate MLP and CNN on CIFAR-10"
)

parser.add_argument(
    "--epochs",
    type=int,
    required=True,
    help="Number of training epochs"
)

parser.add_argument(
    "--lr",
    type=float,
    required=True,
    help="Learning rate"
)

torch.manual_seed(42)

args = parser.parse_args()

EPOCHS = args.epochs
LR = args.lr


# ==================================================
# OUTPUT DIRECTORY
# ==================================================

SAVE_DIR = f"./exp_results/test_met_sgd{EPOCHS}_lr_{LR}"

MODEL_DIR = os.path.join(
    SAVE_DIR,
    "models"
)

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ==================================================
# DEVICE
# ==================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\n" + "=" * 60)
print("EXPERIMENT CONFIGURATION")
print("=" * 60)

print(f"Device:         {device}")
print(f"Epochs:         {EPOCHS}")
print(f"Learning Rate:  {LR}")
print(f"Batch Size:     64")
print(f"Random Seed:    42")
print(f"Results Dir:    {SAVE_DIR}")


# ==================================================
# DATA
# ==================================================

print("\n" + "=" * 60)
print("LOADING DATA")
print("=" * 60)

train_loader, val_loader, test_loader = get_dataloaders(
    batch_size=64,
    seed=42
)

print("Data loaded successfully.")


# ==================================================
# MLP
# ==================================================

print("\n" + "=" * 60)
print("TRAINING MLP")
print("=" * 60)

mlp_model = mlp_arc()

mlp_model, mlp_history, mlp_training_time = train_model(
    mlp_model,
    train_loader,
    val_loader,
    device,
    epochs=EPOCHS,
    learning_rate=LR
)

print("\nMLP training completed.")


# ==================================================
# CNN
# ==================================================

print("\n" + "=" * 60)
print("TRAINING CNN")
print("=" * 60)

cnn_model = cnn_arc()

cnn_model, cnn_history, cnn_training_time = train_model(
    cnn_model,
    train_loader,
    val_loader,
    device,
    epochs=EPOCHS,
    learning_rate=LR
)

print("\nCNN training completed.")


# ==================================================
# FINAL TEST EVALUATION
# ==================================================

print("\n" + "=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)


# --------------------------------------------------
# MLP
# --------------------------------------------------

print("\nEvaluating MLP...")

(
    mlp_test_loss,
    mlp_test_accuracy,
    mlp_precision,
    mlp_recall,
    mlp_f1,
    mlp_cm,
    mlp_per_class,
    mlp_per_class_precision,
    mlp_per_class_recall,
    mlp_per_class_f1
) = evaluate_model(
    mlp_model,
    test_loader,
    device
)


# --------------------------------------------------
# CNN
# --------------------------------------------------

print("\nEvaluating CNN...")

(
    cnn_test_loss,
    cnn_test_accuracy,
    cnn_precision,
    cnn_recall,
    cnn_f1,
    cnn_cm,
    cnn_per_class,
    cnn_per_class_precision,
    cnn_per_class_recall,
    cnn_per_class_f1
) = evaluate_model(
    cnn_model,
    test_loader,
    device
)


# ==================================================
# SAVE VISUALIZATIONS
# ==================================================

print("\n" + "=" * 60)
print("SAVING VISUALIZATIONS")
print("=" * 60)


# --------------------------------------------------
# TRAINING CURVES
# --------------------------------------------------

plot_training_curves(
    mlp_history,
    cnn_history,
    save_dir=SAVE_DIR
)

print("Training curves saved.")


# --------------------------------------------------
# CONFUSION MATRICES
# --------------------------------------------------

plot_confusion_matrices(
    mlp_cm,
    cnn_cm,
    save_dir=SAVE_DIR
)

print("Confusion matrices saved.")


# ==================================================
# SAVE METRICS
# ==================================================

print("\n" + "=" * 60)
print("SAVING METRICS")
print("=" * 60)

save_metrics(
    # MLP
    mlp_test_loss,
    mlp_test_accuracy,
    mlp_precision,
    mlp_recall,
    mlp_f1,
    mlp_per_class,
    mlp_per_class_precision,
    mlp_per_class_recall,
    mlp_per_class_f1,
    mlp_training_time,

    # CNN
    cnn_test_loss,
    cnn_test_accuracy,
    cnn_precision,
    cnn_recall,
    cnn_f1,
    cnn_per_class,
    cnn_per_class_precision,
    cnn_per_class_recall,
    cnn_per_class_f1,
    cnn_training_time,

    save_dir=SAVE_DIR
)

print("Metrics saved.")


# ==================================================
# SAVE TRAINED MODELS
# ==================================================

print("\n" + "=" * 60)
print("SAVING MODELS")
print("=" * 60)

mlp_model_path = os.path.join(
    MODEL_DIR,
    "mlp_model.pth"
)

cnn_model_path = os.path.join(
    MODEL_DIR,
    "cnn_model.pth"
)

torch.save(
    mlp_model.state_dict(),
    mlp_model_path
)

torch.save(
    cnn_model.state_dict(),
    cnn_model_path
)

print(f"MLP model saved: {mlp_model_path}")
print(f"CNN model saved: {cnn_model_path}")


# ==================================================
# FINAL SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)


print(
    f"\nMLP Test Accuracy: "
    f"{mlp_test_accuracy * 100:.2f}%"
)

print(
    f"CNN Test Accuracy: "
    f"{cnn_test_accuracy * 100:.2f}%"
)


print(
    f"\nMLP Test Loss: "
    f"{mlp_test_loss:.4f}"
)

print(
    f"CNN Test Loss: "
    f"{cnn_test_loss:.4f}"
)


print(
    f"\nMLP Macro Precision: "
    f"{mlp_precision * 100:.2f}%"
)

print(
    f"CNN Macro Precision: "
    f"{cnn_precision * 100:.2f}%"
)


print(
    f"\nMLP Macro Recall: "
    f"{mlp_recall * 100:.2f}%"
)

print(
    f"CNN Macro Recall: "
    f"{cnn_recall * 100:.2f}%"
)


print(
    f"\nMLP Macro F1: "
    f"{mlp_f1 * 100:.2f}%"
)

print(
    f"CNN Macro F1: "
    f"{cnn_f1 * 100:.2f}%"
)


print(
    f"\nMLP Training Time: "
    f"{mlp_training_time:.2f}s"
)

print(
    f"CNN Training Time: "
    f"{cnn_training_time:.2f}s"
)


# ==================================================
# OUTPUT LOCATION
# ==================================================

print("\n" + "=" * 60)
print("EXPERIMENT SAVED")
print("=" * 60)

print("\nResults directory:")
print(f"  {SAVE_DIR}/")

print("\nModels:")
print(f"  {mlp_model_path}")
print(f"  {cnn_model_path}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
