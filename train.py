import time

import torch
import torch.nn as nn
import torch.optim as optim


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs=20,
    learning_rate=0.001
):
    """
    Train a classification model.

    Parameters
    ----------
    model : nn.Module
    train_loader : DataLoader
    val_loader : DataLoader
    device : torch.device
    epochs : int
    learning_rate : float

    Returns
    -------
    model : trained model
    history : dict
    training_time : float
    """

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # optimizer = optim.Adam(
    #     model.parameters(),
    #     lr=learning_rate
    # )
    
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    start_time = time.time()

    for epoch in range(epochs):

        # =====================
        # TRAINING
        # =====================

        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            train_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            train_correct += (
                predictions == labels
            ).sum().item()

            train_total += labels.size(0)

        train_loss /= train_total
        train_accuracy = train_correct / train_total

        # =====================
        # VALIDATION
        # =====================

        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)

                predictions = outputs.argmax(dim=1)

                val_correct += (
                    predictions == labels
                ).sum().item()

                val_total += labels.size(0)

        val_loss /= val_total
        val_accuracy = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)

        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        print(
            f"Epoch [{epoch + 1}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

    training_time = time.time() - start_time

    return model, history, training_time
