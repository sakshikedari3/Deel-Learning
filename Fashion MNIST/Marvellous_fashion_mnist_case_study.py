###################################################################################################
# HOW TO RUN THIS :
# Open the terminal and type :
# python -m venv venv
# Then activate it :
# venv\scripts\activate
# Then install libraries:
# pip install tensorflow scikit-learn matplotlib numpy pandas tqdm seaborn
# Press YES in VS Code when prompted
###################################################################################################

###################################################################################################
# Required Libraries
###################################################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import argparse
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import json

###################################################################################################
# Predefined Constants
###################################################################################################

SEED = 42
ARTIFACT_DIR = "artifacts"
BEST_MODEL = os.path.join(ARTIFACT_DIR, "best_model.h5")
FINAL_MODEL = os.path.join(ARTIFACT_DIR, "final_model.h5")
FASHION_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

###################################################################################################
# Function: ensure_dir
# Inputs: path
# Outputs: None
# Description: Check if the directory at the given path exists; if not, create it.
# Author: Sakshi Santosh Kedari
# Date: 19/10/2025
###################################################################################################
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

###################################################################################################
# Function: set_seed
# Inputs: seed (int)
# Outputs: None
# Description: Sets random seed for reproducibility across numpy, random, and TensorFlow.
# Author: Sakshi Santosh Kedari
# Date: 19/10/2025
###################################################################################################
def set_seed(seed=42):
    np.random.seed(seed)
    import random
    random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)

###################################################################################################
# Function: plot_training_curves
# Inputs: history, out_dir
# Outputs: None
# Description: Plots and saves training and validation accuracy/loss curves.
# Author: Sakshi Santosh Kedari
# Date: 19/10/2025
###################################################################################################
def plot_training_curves(history, out_dir):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Val")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Val")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"))
    plt.close()

###################################################################################################
# Function: plot_confusion_matrix
# Inputs: y_true, y_pred, out_dir, normalize
# Outputs: None
# Description: Plots and saves confusion matrix heatmap.
# Author: Sakshi Santosh Kedari
# Date: 19/10/2025
###################################################################################################
def plot_confusion_matrix(y_true, y_pred, out_dir, normalize=False):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                xticklabels=FASHION_CLASSES, yticklabels=FASHION_CLASSES)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
    plt.close()

###################################################################################################
# Function: save_classification_report
# Inputs: y_true, y_pred, out_dir
# Outputs: None
# Description: Saves classification report as JSON.
# Author: Sakshi Santosh Kedari
# Date: 19/10/2025
###################################################################################################
def save_classification_report(y_true, y_pred, out_dir):
    report = classification_report(y_true, y_pred, target_names=FASHION_CLASSES, output_dict=True)
    with open(os.path.join(out_dir, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=2)

###################################################################################################
# Function: show_misclassifications
# Inputs: x, y_true, y_pred, limit, out_dir
# Outputs: None
# Description: Saves grid of misclassified images.
# Author: Sakshi Santosh Kedari
# Date: 19/10/2025
###################################################################################################
def show_misclassifications(x, y_true, y_pred, limit=25, out_dir="artifacts"):
    wrong = np.where(y_true != y_pred)[0]
    sample = wrong[:limit]
    cols = int(np.ceil(np.sqrt(limit)))
    rows = int(np.ceil(limit / cols))
    plt.figure(figsize=(2.8 * cols, 2.8 * rows))
    for i, idx in enumerate(sample):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(x[idx].squeeze(), cmap="gray")
        plt.title(f"P:{FASHION_CLASSES[y_pred[idx]]}\nT:{FASHION_CLASSES[y_true[idx]]}", fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "misclassifications.png"))
    plt.close()

###################################################################################################
# Function: save_label_map
# Inputs: out_dir
# Outputs: None
# Description: Saves label index-to-class mapping as JSON.
# Author: Sakshi Santosh Kedari
# Date: 19/10/2025
###################################################################################################
def save_label_map(out_dir):
    with open(os.path.join(out_dir, "label_map.json"), "w") as f:
        json.dump({i: label for i, label in enumerate(FASHION_CLASSES)}, f, indent=2)

###################################################################################################
# Function: save_summary
# Inputs: acc, loss, epochs, out_dir
# Outputs: None
# Description: Saves test accuracy, loss, and epochs to summary JSON.
# Author: Sakshi Santosh Kedari
# Date: 19/10/2025
###################################################################################################
def save_summary(acc, loss, epochs, out_dir):
    summary = {
        "test_accuracy": round(acc, 4),
        "test_loss": round(loss, 4),
        "epochs_trained": epochs
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

###################################################################################################
# Function: load_data
# Inputs: val_split
# Outputs: Train, validation, test splits
# Description: Loads Fashion-MNIST and splits into train/val/test with normalization.
# Author: Sakshi Santosh Kedari
# Date: 19/10/2025
###################################################################################################
def load_data(val_split=0.1):
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=val_split, random_state=SEED, stratify=y_train
    )
    x_train = (x_train.astype("float32") / 255.0)[..., None]
    x_val = (x_val.astype("float32") / 255.0)[..., None]
    x_test = (x_test.astype("float32") / 255.0)[..., None]
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

###################################################################################################
# Function: load_flattened
# Inputs: None
# Outputs: Flattened train and test arrays
# Description: Loads Fashion-MNIST and flattens images for classical ML models.
# Author: Sakshi Santosh Kedari
# Date: 19/10/2025
###################################################################################################
def load_flattened():
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train = x_train.reshape(len(x_train), -1).astype("float32") / 255.0
    x_test = x_test.reshape(len(x_test), -1).astype("float32") / 255.0
    return x_train, y_train, x_test, y_test

###################################################################################################
# Function: build_cnn
# Inputs: Ir (float) - learning rate
# Outputs: Compiled Keras CNN model
# Description: Builds and compiles CNN with augmentation, Conv2D, Dropout, BatchNorm, Dense.
# Author: Sakshi Santosh Kedari
# Date: 19/10/2025
###################################################################################################
def build_cnn(Ir=1e-3) -> keras.Model:
    data_augmentation = keras.Sequential(
        [
            layers.RandomTranslation(0.05, 0.05, fill_mode="nearest"),
            layers.RandomRotation(0.05, fill_mode="nearest"),
            layers.RandomZoom(0.05, 0.05, fill_mode="nearest"),
        ],
        name="augmentation",
    )
    inputs = keras.Input(shape=(28, 28, 1))
    x = data_augmentation(inputs)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.40)(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="fashion_cnn")
    opt = keras.optimizers.Adam(learning_rate=Ir)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

###################################################################################################
# Function: train_and_evaluate
# Inputs: batch_size (int), epochs (int), Ir (float)
# Outputs: Trained model + saved artifacts
# Description: Trains CNN on Fashion-MNIST, evaluates, saves curves, confusion matrix, misclassifications, report.
# Author: Sakshi Santosh Kedari
# Date: 19/10/2025
###################################################################################################
def train_and_evaluate(batch_size=128, epochs=10, Ir=1e-3):
    ensure_dir(ARTIFACT_DIR)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data(val_split=0.1)
    model = build_cnn(Ir=Ir)
    model.summary()
    callbacks = [
        keras.callbacks.ModelCheckpoint(BEST_MODEL, monitor="val_accuracy", save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True, monitor="val_loss"),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-5, verbose=1),
    ]
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2
    )
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"[TEST] loss: {test_loss:.4f} | acc: {test_acc:.4f}")
    model.save(FINAL_MODEL)
    plot_training_curves(history, ARTIFACT_DIR)
    y_pred = model.predict(x_test, batch_size=256).argmax(axis=1)
    plot_confusion_matrix(y_test, y_pred, ARTIFACT_DIR, normalize=True)
    save_classification_report(y_test, y_pred, ARTIFACT_DIR)
    show_misclassifications(x_test, y_test, y_pred, limit=25, out_dir=ARTIFACT_DIR)
    save_label_map(ARTIFACT_DIR)
    save_summary(test_acc, test_loss, len(history.history["loss"]), ARTIFACT_DIR)

###################################################################################################
# Function: inference_grid
# Inputs: n_samples (int), seed (int)
# Outputs: Saved inference grid image
# Description: Loads saved model, predicts on random test samples, saves predictions grid.
# Author: Sakshi Santosh Kedari
# Date: 19/10/2025
###################################################################################################
def inference_grid(n_samples=9, seed=7):
    if not os.path.exists(BEST_MODEL):
        print(f"Could not find {BEST_MODEL}. Train first with --train.")
        return
    (_, _), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_test = (x_test.astype("float32") / 255.0)[..., None]
    model = keras.models.load_model(BEST_MODEL)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x_test), size=n_samples, replace=False)
    imgs = x_test[idx]
    labs = y_test[idx]
    preds = model.predict(imgs, verbose=0).argmax(axis=1)
    cols = int(np.ceil(np.sqrt(n_samples)))
    rows = int(np.ceil(n_samples / cols))
    plt.figure(figsize=(2.8 * cols, 2.8 * rows))
    for i in range(n_samples):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(imgs[i].squeeze(), cmap="gray")
        plt.title(f"P:{FASHION_CLASSES[preds[i]]}\nT:{FASHION_CLASSES[labs[i]]}", fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    ensure_dir(ARTIFACT_DIR)
    out_path = os.path.join(ARTIFACT_DIR, "inference_grid.png")
    plt.savefig(out_path)
    plt.close()
    print("Saved:", out_path)

###################################################################################################
# Function: run_baselines
# Inputs: None
# Outputs: Prints accuracy of LogReg, LinearSVC, RandomForest
# Description: Runs classical ML baselines on flattened Fashion-MNIST.
# Author: Sakshi Santosh Kedari
# Date: 19/10/2025
###################################################################################################
def run_baselines():
    x_train, y_train, x_test, y_test = load_flattened()
    logreg = Pipeline([
        ("scaler", StandardScaler(with_mean=True)),
        ("clf", LogisticRegression(max_iter=200, n_jobs=-1))
    ])
    logreg.fit(x_train, y_train)
    print("LogReg acc:", accuracy_score(y_test, logreg.predict(x_test)))
    svm = Pipeline([
        ("scaler", StandardScaler(with_mean=True)),
        ("clf", LinearSVC(C=1.0, dual=True, max_iter=5000))
    ])
    svm.fit(x_train, y_train)
    print("LinearSVC acc:", accuracy_score(y_test, svm.predict(x_test)))
    rf = RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1, random_state=SEED)
    rf.fit(x_train, y_train)
    print("RandomForest acc:", accuracy_score(y_test, rf.predict(x_test)))

###################################################################################################
# Function: parse_args
# Inputs: None
# Outputs: Parsed CLI arguments
# Description: Defines command-line arguments for training, inference, baselines.
# Author: Sakshi Santosh Kedari
# Date: 19/10/2025
###################################################################################################
def parse_args():
    p = argparse.ArgumentParser(description="Fashion-MNIST Case Study")
    p.add_argument("--train", action="store_true", help="Train the CNN and save artifacts.")
    p.add_argument("--infer", action="store_true", help="Generate an inference grid using saved model.")
    p.add_argument("--samples", type=int, default=9, help="Number of samples for inference grid.")
    p.add_argument("--baselines", action="store_true", help="Run classical ML baselines.")
    p.add_argument("--epochs", type=int, default=15, help="Epochs for CNN training.")
    p.add_argument("--batch", type=int, default=128, help="Batch size for CNN training.")
    p.add_argument("--Ir", type=float, default=1e-3, help="Learning rate for Adam.")
    return p.parse_args()

###################################################################################################
# Function: main
# Inputs: None
# Outputs: None
# Description: Entry point. Parses CLI args, runs train/infer/baselines accordingly.
# Author: Sakshi Santosh Kedari
# Date: 19/10/2025
###################################################################################################
def main():
    set_seed(SEED)
    ensure_dir(ARTIFACT_DIR)
    args = parse_args()
    did_anything = False
    if args.train:
        train_and_evaluate(batch_size=args.batch, epochs=args.epochs, Ir=args.Ir)
        did_anything = True
    if args.infer:
        inference_grid(n_samples=args.samples)
        did_anything = True
    if args.baselines:
        run_baselines()
        did_anything = True
    if not did_anything:
        print("Nothing to do. Try:\n python fashion_mnist_case")


###################################################################################################
# application starter
###################################################################################################

if __name__ == "__main__" :
    main()
