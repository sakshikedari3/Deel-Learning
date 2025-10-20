import os
import argparse
import random
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# Sklearn for metrics and baselines
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

##############################################################################
# Global config
##############################################################################

SEED = 42
DIGIT_CLASSES = [str(i) for i in range(10)]
ARTIFACT_DIR = "artifacts_digits"
BEST_MODEL = os.path.join(ARTIFACT_DIR, "digits_cnn.h5")
FINAL_MODEL = os.path.join(ARTIFACT_DIR, "digits_cnn_final.h5")

#############################################################################
# Utils
#############################################################################
# # Function: set_seed
# Inputs: seed (int) - random seed value
# Outputs: None
# Description: Sets seed for Python, NumPy, and TensorFlow to ensure reproducibility.
# Author: Sakshi Santosh Kedari
# Date: 20/10/2025
#############################################################################

def set_seed(seed: int = SEED):
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

##############################################################################
# Function: ensure_dir
# Inputs: path (str) - directory path
# Outputs: None
# Description: Creates directory if it does not exist.
# Author: Sakshi Santosh Kedari
# Date: 20/10/2025
################################################################################
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

##############################################################################
# Function: plot_training_curves
# Inputs: history (keras.callbacks.History), out
# Inputs: history (keras.callbacks.History), out
#input : history (keras.callback.History), out_dir (str)
# Outputs: Saves accuracy and loss curve plots
# Description: Plots and saves training/validation accuracy and loss curves.
# Author: Sakshi Santosh Kedari
# Date: 20/10/2025
###############################################################################

def plot_training_curves(history, out_dir=ARTIFACT_DIR):
    ensure_dir(out_dir)
    # Accuracy
    plt.figure(figsize=(7, 5))
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "acc_curve.png"))
    plt.close()
    # Loss
    plt.figure(figsize=(7, 5))
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"))
    plt.close()

##############################################################################
# Function: _annotate_confmat
# Inputs: ax (matplotlib.axes.Axes), cm (np.ndarray), fmt (str)
# Outputs: Annotated confusion matrix
# Description: Adds numeric annotations to confusion matrix heatmap.
# Author: Sakshi Santosh Kedari
# Date: 20/10/2025
###############################################################################

def _annotate_confmat(ax, cm, fmt="{ :. 2f}"):
    n, m = cm.shape
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(m):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            txt = f"{val :.2f}" if isinstance(val, float) else f"{val:d}"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=8)

##################################################################################
# Function: plot_confusion_matrix
# Inputs: y_true (np.ndarray), y_pred (np.ndarray), out_dir (str), normalize (bool)
# Outputs: Saves confusion matrix plot
# Description: Generates and saves confusion matrix as image.
# Author: Sakshi Santosh Kedari
# Date: 20/10/2025
###################################################################################

def plot_confusion_matrix(y_true, y_pred, out_dir=ARTIFACT_DIR, normalize=True):
    ensure_dir(out_dir)
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype("float")/ cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8.5,7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(DIGIT_CLASSES)),
        yticks=np.arange(len(DIGIT_CLASSES)),
        xticklabels=DIGIT_CLASSES,
        yticklabels=DIGIT_CLASSES,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix (normalized)" if normalize else "Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    _annotate_confmat(ax, cm)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix.png"))
    
    plt.close(fig)
    
#####################################################################################
# Function: save_classification_report
# Inputs: y_true (np.ndarray), y_pred (np.ndarray), out_dir (str)
# Outputs: Saves classification report
# Description: Prints and saves classification report with precision, recall, and F1 score.
# Author: Sakshi Santosh Kedari
# Date: 20/10/2025
#################################################################################3##

def save_classification_report(y_true, y_pred, out_dir=ARTIFACT_DIR):
    ensure_dir(out_dir)

    report = classification_report(
    y_true, y_pred, target_names=DIGIT_CLASSES, digits=4
    )
    print(report)
    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write(report)

########################################################################################
# Function: show_misclassifications
# Inputs: x (np.ndarray), y_true (np.ndarray), y_pred (np.ndarray), limit (int), out_dir (str)
# Outputs: Saves misclassified images grid
# Description: Displays and saves examples of misclassified test images.
# Author: Sakshi Santosh Kedari
# Date: 20/10/2025
##########################################################################################

def show_misclassifications(x, y_true, y_pred, limit=25, out_dir=ARTIFACT_DIR):
    ensure_dir(out_dir)
    wrong = np.where(y_true != y_pred)[0]

    if len(wrong) == 0:
        print("No misclassifications")
        return
    sel = wrong[:limit]
    cols = 5
    rows = int(np.ceil(len(sel) / cols))
    plt.figure(figsize=(12,2.6 * rows))
    for i, idx in enumerate(sel, 1):
        img = x[idx].squeeze()
        plt.subplot(rows, cols, i)
        plt.imshow(img, cmap="gray")
        plt.title(f"T: {DIGIT_CLASSES[y_true[idx]]}\nP:{DIGIT_CLASSES[y_pred[idx]]}", fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "misclassifications.png"))
    plt.close()

##########################################################################################
# Function: save_label_map
# Inputs: out_dir (str)
# Outputs: Saves label mapping file
# Description: Saves numeric-to-class name mapping (0-9) for MNIST digits.
# Author: Sakshi Santosh Kedari
# Date: 20/10/2025
###########################################################################################

def save_label_map(out_dir=ARTIFACT_DIR):
    ensure_dir(out_dir)
    with open(os.path.join(out_dir, "label_map.txt"), "w") as f:
        for i, name in enumerate(DIGIT_CLASSES):
            f.write(f"{i}: {name}\n")

########################################################################################
# Function: save_summary
# Inputs: test acc (float), test_loss (float), epochs (int), out_dir (str)
# Outputs: Saves summary text file
# Description: Saves final model performance summary and artifact list.
# Author: Sakshi Santosh Kedari
# Date: 20/10/2025
#######################################################################################

def save_summary(test_acc, test_loss, epochs, out_dir=ARTIFACT_DIR):
    ensure_dir(out_dir)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(
            "MNIST Digits CNN Summary\n"
            f"Test Accuracy: {test_acc :.4f}\n"
            f"Test Loss : {test_loss :.4f}\n"
            f"Epochs : {epochs}\n"
            "Artifacts : acc_curve.png, loss_curve.png, confusion_matrix.png,\n"
            "           misclassifications.png, classification_report.txt,\n"
            "           label_map.txt, digits_cnn.h5, digits_cnn_final.h5\n"
        )

##########################################################################################
# Function: load_data
# Inputs: val_split (float) - validation split ratio
# Outputs: (train_data, val_data, test_data) tuples
# Description: Loads MNIST dataset, normalizes and splits into train, val, test sets.
# Author: Sakshi Santosh Kedari
# Date: 20/10/2025
#########################################################################################

def load_data(val_split=0.1) -> Tuple[Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray]]:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=val_split, random_state=SEED, stratify=y_train
    )
    # Scale and add channel dim
    x_train = (x_train.astype("float32") / 255.0)[ ... , None]
    x_val =(x_val.astype("float32") / 255.0)[ ... , None]
    x_test =(x_test.astype("float32") / 255.0)[ ... , None]
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

#######################################################################################
# Function: load_flattened
# Inputs: None
# Outputs: Flattened train and test arrays with labels
# Description: Loads MNIST and flattens images for classical ML models.
# Author: Sakshi Santosh Kedari
# Date: 20/10/2025
#######################################################################################

def load_flattened():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(len(x_train), -1).astype("float32") / 255.0
    x_test = x_test.reshape(len(x_test), -1).astype("float32") / 255.0
    return x_train, y_train, x_test, y_test

######################################################################################
# Function: build_cnn
# Inputs: Ir (float) - learning rate
# Outputs: Compiled Keras CNN model
# Description: Builds and compiles CNN with augmentation, Conv2D, Dropout, BatchNorm, Dense.
# Author: Sakshi Santosh Kedari
# Date: 20/10/2025
########################################################################################

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
    model = keras.Model(inputs, outputs, name="digits_cnn")
    opt = keras.optimizers.Adam(learning_rate=Ir)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

##################################
# Train / Evaluate
##################################:

########################################################################################
# Function: train_and_evaluate
# Inputs: batch_size (int), epochs (int), Ir (float)
# Outputs: Trained model + saved artifacts
# Description: Trains CNN on MNIST, evaluates, saves curves, confusion matrix, misclassifications
# Author: Sakshi Santosh Kedari
# Date: 20/10/2025
#######################################################################################

def train_and_evaluate(batch_size=128, epochs=10, Ir=1e-3):
    ensure_dir(ARTIFACT_DIR)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data(val_split=0.1)
    model = build_cnn(Ir=Ir)
    model.summary()
    callbacks = [
        keras.callbacks.ModelCheckpoint(BEST_MODEL, monitor="val_accuracy", save_best_only=True, verbose=1),
    keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_loss"),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2, min_Ir=1e-5, verbose=1),
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
    print(f"[TEST] loss: {test_loss :.4f} | acc: {test_acc :.4f}")
    model.save(FINAL_MODEL)
    # Metrics & artifacts
    plot_training_curves(history, ARTIFACT_DIR)
    y_pred = model.predict(x_test, batch_size=256).argmax(axis=1)
    plot_confusion_matrix(y_test, y_pred, ARTIFACT_DIR, normalize=True)
    save_classification_report(y_test, y_pred, ARTIFACT_DIR)
    show_misclassifications(x_test, y_test, y_pred, limit=25, out_dir=ARTIFACT_DIR)
    save_label_map(ARTIFACT_DIR)
    save_summary(test_acc, test_loss, len(history.history["loss"]), ARTIFACT_DIR)

####################################################################################
# Function: inference_grid
# Inputs: n_samples (int), seed (int)
# Outputs: Saved inference grid image
# Description: Loads saved model, predicts on random test samples, saves predictions grid.
# Author: Sakshi Santosh Kedari
# Date: 20/10/2025
######################################################################################

def inference_grid(n_samples=9, seed=7):
    if not os.path.exists(BEST_MODEL):
        print(f"Could not find {BEST_MODEL}. Train first with -- train.")
        return
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test =(x_test.astype("float32") / 255.0)[ ... , None]
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
        plt.title(f"P:{DIGIT_CLASSES[preds[i]]}\nT:{DIGIT_CLASSES[labs[i]]}", fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    ensure_dir(ARTIFACT_DIR)
    out_path = os.path.join(ARTIFACT_DIR, "inference_grid.png")
    plt.savefig(out_path)
    plt.close()
    print("Saved:", out_path)

##############################################################################3
# Baselines
###############################################################################
# Function: run_baselines
# Inputs: None
# Outputs: Prints accuracy of LogReg, LinearSVC, RandomForest
# Description: Runs classical ML baselines on flattened MNIST digits.
# Author: Sakshi Santosh Kedari
# Date: 20/10/2025
###############################################################################

def run_baselines():
    x_train, y_train, x_test, y_test = load_flattened()
    # Logistic Regression
    logreg = Pipeline([
    ("scaler", StandardScaler(with_mean=True)),
    ("clf", LogisticRegression(max_iter=200, n_jobs =- 1))
    ])
    logreg.fit(x_train, y_train)
    print("LogReg acc:", accuracy_score(y_test, logreg.predict(x_test)))
    # Linear SVM
    svm = Pipeline([
        ("scaler", StandardScaler(with_mean=True)),
        ("clf", LinearSVC(C=1.0, dual=True, max_iter=5000))
        ])
    svm.fit(x_train, y_train),
    print("LinearSVC acc:", accuracy_score(y_test, svm.predict(x_test)))
    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs =- 1,
    random_state=SEED)
    rf.fit(x_train, y_train)
    print("RandomForest acc:", accuracy_score(y_test, rf.predict(x_test)))

###################################################################################
# Function: parse_args
# Inputs: None
# Outputs: Parsed CLI arguments
# Description: Defines command-line arguments for training, inference, baselines.
# Author: Sakshi Santosh Kedari
# Date: 20/10/2025
#####################################################################################

def parse_args(): 
        p = argparse.ArgumentParser(description="MNIST Digits Classification Case Study")
        p.add_argument("--train", action="store_true", help="Train the CNN and save artifacts.")
        p.add_argument("--infer", action="store_true", help="Generate an inference grid using saved model.")
        p.add_argument("--samples", type=int, default=9, help="Number of samples for inference grid.")
        p.add_argument("--baselines", action="store_true", help="Run classical ML baselines.")
        p.add_argument("--epochs", type=int, default=10, help="Epochs for CNN training.")
        p.add_argument("--batch", type=int, default=128, help="Batch size for CNN training.")
        p.add_argument("--Ir", type=float, default=1e-3, help="Learning rate for Adam.")
        return p.parse_args()

######################################################################################
# Function: main
# Inputs: None
# Outputs: None
# Description: Entry point. Parses CLI args, runs train/infer/baselines accordingly.
# Author: Sakshi Santosh Kedari
# Date: 20/10/2025
######################################################################################3
def main():
    set_seed(SEED)
    ensure_dir(ARTIFACT_DIR)
    args = parse_args()
    print(args)
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
        print(
            "Nothing to do. Try one of:\n"
            "python digits_classification_case_study.py -- train\n"
            "python digits_classification_case_study.py -- infer -- samples 9\n"
            "python digits_classification_case_study.py -- baselines\n"
    )

##############################################################################
#stater
###############################################################################

if __name__ == "__main__" :
    main()