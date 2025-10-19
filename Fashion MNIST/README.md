## 🧵 Fashion-MNIST Case Study

A complete machine learning pipeline for image classification using both deep learning (CNN) and classical ML models. Built for reproducibility, modularity, and recruiter visibility.

---

### 📦 Project Structure

```
fashion_mnist_case_study.py
artifacts/
├── best_model.h5
├── final_model.h5
├── training_curves.png
├── confusion_matrix.png
├── misclassifications.png
├── inference_grid.png
├── classification_report.json
├── summary.json
├── label_map.json
```

---

### 🚀 How to Run

#### 1. Setup Environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # macOS/Linux

pip install tensorflow scikit-learn matplotlib numpy pandas tqdm seaborn
```

#### 2. Train CNN Model

```bash
python fashion_mnist_case_study.py --train --epochs 15 --batch 128 --Ir 0.001
```

#### 3. Run Inference Grid

```bash
python fashion_mnist_case_study.py --infer --samples 9
```

#### 4. Run Classical ML Baselines

```bash
python fashion_mnist_case_study.py --baselines
```

---

### 🧠 Features

- ✅ CNN with data augmentation, dropout, batch normalization
- ✅ Classical ML baselines: Logistic Regression, Linear SVM, Random Forest
- ✅ Artifact logging: models, plots, reports, misclassifications
- ✅ CLI interface for training, inference, and evaluation
- ✅ Reproducible results with fixed random seed

---

### 📊 Outputs

- `training_curves.png`: Accuracy and loss over epochs
- `confusion_matrix.png`: Normalized confusion matrix
- `misclassifications.png`: Grid of misclassified test samples
- `inference_grid.png`: Predictions on random test samples
- `classification_report.json`: Precision, recall, F1-score per class
- `summary.json`: Final test accuracy, loss, and epochs
- `label_map.json`: Index-to-class mapping

---

### 🛠 Technologies Used

- Python 3.10+
- TensorFlow / Keras
- scikit-learn
- matplotlib, seaborn
- argparse, json

---

### 👩‍💻 Author

**Sakshi Santosh Kedari**  
Master’s student in Computer Science | Data Science & ML  
Actively seeking Python developer roles in Pune  
📫 [LinkedIn](https://www.linkedin.com) | 🧠 [GitHub](https://github.com)


