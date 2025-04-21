# Gradient Boosting Tree Classifier

This repository contains an implementation of a **Gradient Boosting Tree Classifier**, built from scratch using **NumPy**. The model is designed for **binary classification**, combining multiple **decision trees** as weak learners to iteratively minimize the **log-loss** via gradient descent. The project includes a **custom dataset generator**, **comprehensive tests** (including a comparison with **scikit-learn**), and an **evaluation script** with a **classification report**.

## Directory Structure

- `model/model.py`: Contains the `GradientBoostingClassifier` and helper `DecisionTree` classes.
- `tests/test_gradient.py`: **Pytest** suite to validate model functionality and compare with **scikit-learn**.
- `tests/generate_dataset.py`: Script to generate a synthetic binary classification dataset (`data/dataset.csv`).
- `tests/evaluate_model.py`: Script to train the model on the dataset and produce a **classification report**.
- `requirements.txt`: Lists dependencies (`numpy`, `pytest`, `scikit-learn`).

## Setup Instructions

1. **Create a Virtual Environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run Tests**:

   Navigate to the project root and run **pytest**:

   ```bash
   cd /path/to/GradientBoosting
   pytest tests/
   ```

4. **Generate Dataset**:

   Create a synthetic dataset for evaluation:

   ```bash
   python tests/generate_dataset.py
   ```

   This generates `tests/dataset.csv` with **1000 samples**, **two features**, and **binary labels**.

5. **Evaluate Model**:

   Train the model and view the **classification report**:

   ```bash
   python tests/evaluate_model.py
   ```

   Outputs **precision**, **recall**, **F1-score**, **accuracy**, and **support** for each class.

## What Does the Model You Have Implemented Do and When Should It Be Used?

### What It Does

The `GradientBoostingClassifier` performs **binary classification** by building an ensemble of **decision trees**, each fitted to the negative gradient of the **log-loss**. It:

- Initializes with a constant **log-odds** prediction.
- Iteratively adds trees that correct errors in the current predictions, scaled by a **learning rate**.
- Outputs **class probabilities** (via sigmoid) or **binary labels** (0 or 1).

The model balances **predictive accuracy** and **robustness** by combining weak learners into a strong classifier.

### When to Use It

Use this model when:

- **Non-linear Relationships**: Your data has complex, non-linear patterns (e.g., **customer churn**, **fraud detection**).
- **High Accuracy**: You need a robust classifier for structured data with sufficient samples and features.
- **Interpretability**: You want to understand **feature importance** through tree splits.


It’s less suitable for **very small datasets** (prone to overfitting) or **real-time applications** (due to training time) compared to simpler models like **logistic regression**.

## How Did You Test Your Model to Determine If It Is Working Reasonably Correctly?

The model’s correctness is validated through a comprehensive test suite in `tests/test_gradient.py`, using **pytest**. Key tests include:

1. **Fit and Predict (`test_gradient_boosting_fit_predict`)**:

   - Trains on a synthetic dataset (**100 samples**, **2 features**, linear boundary).
   - Asserts predictions are **binary** (0 or 1) and **accuracy > 0.7**, ensuring basic functionality.

2. **Predict Probabilities (`test_gradient_boosting_proba`)**:

   - Verifies **probabilities** sum to 1 and are between 0 and 1 for a **50-sample** dataset.

3. **Invalid Parameters (`test_invalid_parameters`)**:

   - Checks that invalid inputs (e.g., `n_estimators=0`, `learning_rate=-0.1`) raise `ValueError`.

4. **Single-Class Data (`test_single_class_data`)**:

   - Ensures the model raises an error for non-binary classification (e.g., all zeros).

5. **Unfitted Model (`test_unfitted_model`)**:

   - Confirms that calling `predict` before `fit` raises an error.

6. **Scikit-learn Comparison (`test_compare_with_sklearn`)**:

   - Trains both `GradientBoostingClassifier` and `sklearn.ensemble.GradientBoostingClassifier` on a **200-sample** synthetic dataset.
   - Asserts **accuracies** are within **3%** (e.g., custom: 0.85, sklearn: 0.87), validating performance against a trusted baseline.

Additionally, the model was tested on a **custom dataset** (`tests/dataset.csv`, **1000 samples**) via `tests/evaluate_model.py`, which produces a **classification report** (**precision**, **recall**, **F1-score**, **accuracy**, **support**). got **accuracy** of ~**0.91**, with balanced metrics for both classes, confirming robustness on larger data.

![image](https://github.com/user-attachments/assets/d250eb56-491a-48b8-ac19-005754abcc56)


These tests cover **functionality**, **edge cases**, **parameter validation**, and **external benchmarking**, ensuring the model performs as expected.

## What Parameters Have You Exposed to Users to Tune Performance?

The `GradientBoostingClassifier` exposes the following parameters for tuning:

- **`n_estimators` (default: 100)**:

  - **Description**: Number of **decision trees** in the ensemble.
  - **Tuning**: Increase (e.g., 200) for better **accuracy** but longer training; decrease (e.g., 10) for faster training with potential **underfitting**.

- **`learning_rate` (default: 0.1)**:

  - **Description**: Step size for updating predictions per tree.
  - **Tuning**: Smaller values (e.g., 0.01) improve **robustness** but require more trees; larger values (e.g., 0.5) speed up training but risk **overshooting**.

- **`max_depth` (default: 3)**:

  - **Description**: Maximum depth of each **decision tree**.
  - **Tuning**: Deeper trees (e.g., 5) capture complex patterns but risk **overfitting**; shallower trees (e.g., 2) are simpler and generalize better.

- **`min_samples_split` (default: 2)**:

  - **Description**: Minimum samples required to split a tree node.
  - **Tuning**: Higher values (e.g., 5) prevent **overfitting** by limiting tree complexity; lower values allow more splits.

### Basic Usage Examples

```python
import numpy as np
from model.model import GradientBoostingClassifier

# Load custom dataset
data = np.loadtxt("tests/dataset.csv", delimiter=",", skiprows=1)
X, y = data[:, :2], data[:, 2].astype(int)

# Initialize and train model
model = GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, max_depth=3)
model.fit(X, y)

# Predict and compute accuracy
predictions = model.predict(X)
accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy:.4f}")

# Predict probabilities
proba = model.predict_proba(X)
print(f"Class probabilities:\n{proba[:5]}")
```

Run evaluation with **classification report**:

```bash
python tests/evaluate_model.py
```

## Are There Specific Inputs That Your Implementation Has Trouble With? Given More Time, Could You Work Around These or Is It Fundamental?

### Trouble Spots

1. **Single-Class Data**:

   - **Issue**: The model raises an error for datasets with one class (e.g., all zeros) due to undefined **log-odds**.
   - **Cause**: **Binary classification** assumes two classes for **log-loss** computation.
   - **Workaround**: Support **multi-class classification** with **softmax** and **cross-entropy loss**, or handle single-class cases with a warning.

2. **Small Datasets**:

   - **Issue**: With few samples (e.g., <20), trees **overfit**, leading to poor generalization (e.g., **accuracy < 0.7** in tests).
   - **Cause**: Limited data restricts tree diversity and gradient updates.
   - **Workaround**: Implement **early stopping** based on validation loss or increase `min_samples_split`.


## Notes

- Run **pytest** from the project root (`GradientBoosting/`) to avoid import errors, or use the `sys.path` adjustment in `test_gradient.py`.
- The **custom dataset** (`tests/dataset.csv`) is designed for **binary classification**; modify `generate_dataset.py` for different distributions.


