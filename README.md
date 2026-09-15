# CIFAR-10 MLP vs CNN Comparison

<h2>🎥 Project Explanation</h2>

<video src="./video_explaination.mp4" controls width="800">
</video>

## 1. Project Overview

This project investigates the performance of **Multi-Layer Perceptrons (MLPs)** and **Convolutional Neural Networks (CNNs)** on the CIFAR-10 image-classification dataset.

The main purpose was to understand how different **neural-network architectures** and **optimization methods** affect classification performance. The project compares:

* MLP vs CNN
* Simple vs Complex architectures
* Adam vs SGD optimizers

A total of **8 models** were trained and evaluated using four architecture/optimizer combinations.

The project also includes a separate evaluation pipeline that loads previously trained models and compares them without retraining.

---

# 2. Objectives

The project was designed to investigate the following questions:

1. Do CNNs perform better than MLPs on image-classification tasks?
2. Does increasing the complexity of the architecture improve performance?
3. How does Adam compare with SGD?
4. How do different models perform across the individual CIFAR-10 classes?
5. Is accuracy alone sufficient to compare the models?

The goal was therefore not only to train models, but also to perform a systematic comparison using multiple evaluation metrics and visualizations.

---

# 3. Dataset

The project uses the **CIFAR-10 dataset**.

CIFAR-10 contains 10 image classes:

* Airplane
* Automobile
* Bird
* Cat
* Deer
* Dog
* Frog
* Horse
* Ship
* Truck

The project uses a data-loading and splitting pipeline implemented in `dataset/dataloader.py`.

For the current comparison notebook, the evaluation uses a **600-sample test set/subset**.

---

# 4. Project Structure

```text
project/
│
├── main.py
│
├── train.py
├── metrics.py
├── metrics2.py
│
├── architecture/
│   ├── mlparc.py
│   └── convnn.py
│
├── dataset/
│   └── dataloader.py
│
├── data/
│   └── split/
│
├── results/
│   └── ...
│
└── cifar10_4_architectures_visualization_with_metrics2.ipynb
```

---

# 5. Role of Each File

## `main.py`

`main.py` is the **main experiment/training driver**.

It connects the different components of the project. It:

1. Loads the dataset.
2. Creates the MLP and CNN models.
3. Calls the training function from `train.py`.
4. Evaluates the trained models.
5. Generates/saves evaluation results.
6. Saves the trained model checkpoints (`.pth` files).

Therefore, `main.py` is the file that coordinates the complete experiment.

---

## `train.py`

`train.py` contains the **actual model-training logic**.

The training function handles operations such as:

* Forward propagation
* Loss calculation
* Backpropagation
* Optimizer updates
* Epoch-based training
* Training/validation loss tracking

The models are trained by calling the training function from this file.

---

## `metrics.py`

`metrics.py` is a **utility file for evaluation and visualization**.

It is not the main training script.

It provides functions for tasks such as:

* Evaluating model performance
* Calculating classification metrics
* Plotting training/validation curves
* Creating confusion matrices
* Saving metrics

`main.py` calls these functions after training to analyze the trained models.

---

## `metrics2.py`

`metrics2.py` was created specifically for **evaluation of already-trained models**.

Unlike the training pipeline, it does **not train the models again**.

It:

1. Loads an existing `.pth` checkpoint.
2. Reconstructs the appropriate architecture.
3. Evaluates the model on the test data.
4. Calculates:

   * Test loss
   * Accuracy
   * Macro precision
   * Macro recall
   * Macro F1-score
   * Confusion matrix
   * Per-class metrics
5. Returns the results for comparison.

This allows the Jupyter notebook to compare all previously trained models without repeating the training process.

---

## `architecture/`

The `architecture` directory contains the model definitions.

It includes the MLP and CNN implementations and their different architecture configurations.

The project uses:

* Simple MLP
* Complex MLP
* Simple CNN
* Complex CNN

---

## `dataset/dataloader.py`

This file contains the data-loading functionality.

It is responsible for preparing the CIFAR-10 data and creating the required data loaders for training, validation, and testing.

---

# 6. Experimental Design

Four main configurations were tested:

| Architecture | Optimizer |
| ------------ | --------- |
| Simple       | Adam      |
| Complex      | Adam      |
| Simple       | SGD       |
| Complex      | SGD       |

Each configuration was tested using both an MLP and a CNN.

Therefore:

```text
4 configurations × 2 model types = 8 models
```

The purpose was to keep the comparison structured and determine the effect of architecture and optimizer choices.

---

# 7. Training Pipeline

The complete training workflow is:

```text
CIFAR-10 Dataset
       ↓
Data Loading
       ↓
Architecture Selection
       ↓
MLP / CNN
       ↓
Optimizer Selection
       ↓
Adam / SGD
       ↓
Training
       ↓
Validation
       ↓
Model Checkpoint (.pth)
       ↓
Evaluation
       ↓
Metrics + Visualizations
```

The trained models are saved so that they can later be evaluated without retraining.

---

# 8. Evaluation-Only Pipeline

After training, the saved checkpoints can be evaluated separately.

The evaluation workflow is:

```text
Existing .pth Models
       ↓
metrics2.py
       ↓
Load Model
       ↓
Test Dataset
       ↓
Calculate Metrics
       ↓
Jupyter Notebook
       ↓
Compare All 8 Models
```

This separation is useful because changing the comparison or generating additional metrics does not require training the models again.

---

# 9. Setup

## Requirements

The project requires Python and the following major libraries:

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn jupyter
```

---

## Dataset Setup

The CIFAR-10 dataset is loaded through the project's data-loading pipeline.

The required project directories should be maintained:

```text
data/
└── split/
```

The dataset preparation/loading code is handled by:

```text
dataset/dataloader.py
```

---

# 10. Running the Project

## Step 1 — Train the Models

Run the main experiment:

```bash
python main.py
```

The main script coordinates model creation, training, evaluation, and saving the results.

If the particular version of `main.py` exposes command-line parameters such as epochs and learning rate, they can be supplied according to the script configuration.

For example:

```bash
python main.py --epochs 30 --lr 0.001
```

---

## Step 2 — Saved Models

After training, model checkpoints are stored as `.pth` files inside the results directories.

These files contain the trained model parameters and can be reused for evaluation.

---

## Step 3 — Evaluate Existing Models

Once the models have already been trained, `metrics2.py` can be used to evaluate them.

**No additional training is performed by `metrics2.py`.**

This makes it possible to calculate accuracy, precision, recall, F1-score, and other evaluation information directly from the saved checkpoints.

---

## Step 4 — Run the Jupyter Notebook

Start Jupyter:

```bash
jupyter notebook
```

Then open:

```text
cifar10_4_architectures_visualization_with_metrics2.ipynb
```

The notebook:

* Loads the existing `.pth` models.
* Uses `metrics2.py`.
* Evaluates all 8 models.
* Creates a comparison table.
* Displays model-performance comparisons.
* Allows confusion matrices and per-class performance to be examined.

The notebook does **not retrain the models**.

---

# 11. Evaluation Metrics

The project uses several metrics to provide a more complete view of model performance.

### Accuracy

Accuracy measures the overall percentage of test samples classified correctly.

### Precision

Precision measures how often the model's predictions for a particular class are actually correct.

### Recall

Recall measures how effectively the model identifies the actual samples belonging to a class.

### F1-Score

F1-score combines precision and recall into a single metric.

### Macro Average

Precision, recall, and F1-score are calculated using **macro averaging**.

Macro averaging gives equal importance to each of the 10 CIFAR-10 classes, rather than allowing classes with more successful predictions to dominate the overall score.

These metrics were selected because accuracy alone does not provide enough information about how the model performs across individual classes.

---

# 12. Results

The current evaluation of the 8 trained models produced the following results:

| Setting        | Model | Test Loss |   Accuracy | Precision | Recall |         F1 |
| -------------- | ----- | --------: | ---------: | --------: | -----: | ---------: |
| Complex + Adam | CNN   |    2.9986 | **56.67%** |    56.62% | 56.67% | **56.42%** |
| Simple + Adam  | CNN   |    3.1685 |     53.00% |    53.03% | 53.00% |     52.89% |
| Complex + Adam | MLP   |    4.4116 |     41.83% |    41.77% | 41.83% |     41.06% |
| Simple + SGD   | MLP   |    1.7688 |     39.83% |    39.51% | 39.83% |     38.99% |
| Simple + Adam  | MLP   |    2.9608 |     38.67% |    38.65% | 38.67% |     38.41% |
| Complex + SGD  | MLP   |    1.9247 |     34.00% |    36.21% | 34.00% |     31.68% |
| Complex + SGD  | CNN   |    1.9399 |     33.50% |    32.27% | 33.50% |     31.02% |
| Simple + SGD   | CNN   |    2.0356 |     28.17% |    24.09% | 28.17% |     23.58% |

---

# 13. Best Performing Model

The best-performing model was the:

**Complex CNN + Adam**

Its performance was:

```text
Accuracy       : 56.67%
Macro Precision: 56.62%
Macro Recall   : 56.67%
Macro F1       : 56.42%
Test Loss      : 2.9986
```

This was the strongest configuration among the eight models evaluated.

---

# 14. What Was Compared?

The project provides three major comparisons.

### MLP vs CNN

The models were compared to determine which type of architecture is better suited to CIFAR-10 image classification.

The results show that the CNN configurations generally performed better than the MLP configurations.

### Simple vs Complex

The project compares simpler and more complex network architectures to investigate whether additional architectural complexity improves classification performance.

The Complex CNN with Adam performed better than the Simple CNN with Adam.

### Adam vs SGD

Both Adam and SGD were tested to investigate the effect of optimizer choice.

Adam generally produced stronger results in these experiments, particularly for the CNN models, although optimizer performance also depended on the architecture.

---

# 15. Visual Analysis

In addition to numerical metrics, the project includes visual analysis.

## Training and Validation Curves

Training curves can be used to examine:

* How the loss changes during training.
* Whether the model continues improving.
* Differences between training and validation performance.
* Potential signs of overfitting.

## Confusion Matrices

Confusion matrices show how predictions are distributed between the ten CIFAR-10 classes.

They help identify:

* Which classes are classified well.
* Which classes are frequently confused.
* Where a particular architecture struggles.

## Per-Class Metrics

Per-class accuracy, precision, recall, and F1-score provide more detailed information than overall accuracy.

This is particularly useful because a model can achieve reasonable overall accuracy while still performing poorly on specific classes.

---

# 16. What Was Built

The final project consists of a complete experimental framework that can:

* Load CIFAR-10 data.
* Create multiple MLP and CNN architectures.
* Train models using Adam and SGD.
* Compare Simple and Complex architectures.
* Save trained model checkpoints.
* Evaluate trained models.
* Calculate multiple classification metrics.
* Generate confusion matrices.
* Plot training and validation curves.
* Analyze individual class performance.
* Compare all trained models in a Jupyter notebook.
* Re-evaluate saved models without retraining.

The project therefore separates **model development/training** from **model evaluation and comparison**.

---

# 17. Planned Approach

The original plan was to build multiple neural-network models and conduct a controlled comparison rather than training only one model.

The intended experimental process was:

1. Prepare the CIFAR-10 dataset.
2. Implement MLP and CNN architectures.
3. Create Simple and Complex versions.
4. Train each architecture with Adam and SGD.
5. Save the trained models.
6. Evaluate the models using multiple metrics.
7. Compare their performance.
8. Use confusion matrices and per-class metrics for deeper analysis.
9. Determine which architecture and optimizer combination performs best.

This approach was implemented through the training pipeline and the evaluation notebook.

---

# 18. Outcome

The experiments successfully produced and compared **8 trained models**.

The results demonstrate that the choice of architecture has a substantial effect on CIFAR-10 classification performance. CNNs generally performed better than MLPs, with the Complex CNN using Adam achieving the highest accuracy.

The project also demonstrates why using multiple metrics is useful. Accuracy provides the overall result, while precision, recall, F1-score, confusion matrices, and per-class metrics provide additional information about how the models behave across different CIFAR-10 categories.

---

# 19. Conclusion

The project demonstrates that **CNNs are better suited to this CIFAR-10 image-classification task than the tested MLP architectures**. Among the eight configurations, the **Complex CNN trained with Adam** achieved the best performance with **56.67% accuracy** and **56.42% macro F1-score**.

The experiments also show that increasing model complexity can improve performance, particularly for CNNs, and that optimizer choice can significantly influence the results. However, the effect of an optimizer is not independent of the architecture, as some MLP configurations showed different behavior.

Overall, the project achieved its main objective of building a controlled framework for comparing **architecture type, architecture complexity, and optimizer choice**, while also providing detailed quantitative and visual analysis of model performance.

---

# 20. Summary

```text
Dataset
  ↓
CIFAR-10
  ↓
┌─────────────────────────────┐
│      Model Architectures    │
│                             │
│  Simple MLP    Complex MLP  │
│  Simple CNN    Complex CNN  │
└─────────────────────────────┘
  ↓
┌──────────────────────┐
│      Optimizers      │
│                      │
│      Adam / SGD      │
└──────────────────────┘
  ↓
8 Trained Models
  ↓
Saved .pth Checkpoints
  ↓
metrics2.py
  ↓
Accuracy / Precision / Recall / F1
  ↓
Confusion Matrices + Per-Class Metrics
  ↓
Jupyter Notebook Comparison
  ↓
Best Model:
Complex CNN + Adam
56.67% Accuracy
```

**Final outcome:** a complete CIFAR-10 experimentation and comparison pipeline for studying the effect of **MLP vs CNN, Simple vs Complex architectures, and Adam vs SGD**.
