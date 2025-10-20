# 🧠 MNIST Digits Classification Case Study

A complete machine learning pipeline for classifying handwritten digits using a Convolutional Neural Network (CNN) and classical ML baselines. Built with TensorFlow, scikit-learn, and Python best practices.

## 📦 Project Structure

```
digits_classification_case_study.py
artifacts_digits/
├── acc_curve.png
├── loss_curve.png
├── confusion_matrix.png
├── misclassifications.png
├── classification_report.txt
├── label_map.txt
├── summary.txt
├── digits_cnn.h5
├── digits_cnn_final.h5
```

## 🚀 Features

- CNN with augmentation, dropout, batch norm, and Adam optimizer
- Early stopping, checkpointing, and learning rate scheduling
- Classical ML baselines: Logistic Regression, Linear SVM, Random Forest
- Artifact logging: plots, reports, model weights, label maps
- CLI interface for training, inference, and baseline comparison
- Inference grid for visual inspection of predictions

## 🧪 Performance Summary

From `summary.txt`:
```
Test Accuracy: 0.9950
Test Loss    : 0.0162
Epochs       : 10
```

From `classification_report.txt`:
- Precision, Recall, F1-score all ≈ 0.995 across classes
- Macro and weighted averages: 0.9950

## 🖼️ Sample Artifacts

- 📈 Accuracy & Loss Curves: `acc_curve.png`, `loss_curve.png`
- 🔍 Confusion Matrix: `confusion_matrix.png`
- ❌ Misclassifications: `misclassifications.png`
- 📊 Classification Report: `classification_report.txt`
- 🔤 Label Map: `label_map.txt`
- 🧠 Saved Models: `digits_cnn.h5`, `digits_cnn_final.h5`

## 🧰 Requirements

- Python 3.8+
- TensorFlow 2.x
- scikit-learn
- matplotlib
- numpy

Install dependencies:
```bash
pip install -r requirements.txt
```

## ⚙️ Usage

### Train CNN and save artifacts
```bash
python digits_classification_case_study.py --train --epochs 10 --batch 128 --Ir 0.001
```

### Run inference grid
```bash
python digits_classification_case_study.py --infer --samples 9
```

### Run classical ML baselines
```bash
python digits_classification_case_study.py --baselines
```

## 📚 Author

**Sakshi Santosh Kedari**  
Master’s in Computer Science | Data Science & ML  
📍 Pune, India  
🔗 [LinkedIn](https://www.linkedin.com/in/sakshi-kedari) | [GitHub](https://github.com/sakshikedari)
