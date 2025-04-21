import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier as SklearnGradientBoostingClassifier

def test_gradient_boosting_fit_predict():
    # Generate synthetic binary classification data
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    model = GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, max_depth=3)
    model.fit(X, y)
    predictions = model.predict(X)
    
    # Check that predictions are binary
    assert set(predictions).issubset({0, 1})
    # Check accuracy is reasonable (should be > 0.7 for this simple data)
    accuracy = np.mean(predictions == y)
    assert accuracy > 0.9

def test_gradient_boosting_proba():
    X = np.random.randn(50, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    model = GradientBoostingClassifier(n_estimators=5, learning_rate=0.1, max_depth=2)
    model.fit(X, y)
    proba = model.predict_proba(X)
    
    # Check probabilities sum to 1
    assert np.allclose(proba.sum(axis=1), 1)
    # Check probabilities are between 0 and 1
    assert np.all((proba >= 0) & (proba <= 1))

def test_invalid_parameters():
    with pytest.raises(ValueError):
        GradientBoostingClassifier(n_estimators=0)
    with pytest.raises(ValueError):
        GradientBoostingClassifier(learning_rate=-0.1)
    with pytest.raises(ValueError):
        GradientBoostingClassifier(max_depth=0)
    with pytest.raises(ValueError):
        GradientBoostingClassifier(min_samples_split=1)

def test_single_class_data():
    X = np.random.randn(20, 2)
    y = np.zeros(20)
    
    model = GradientBoostingClassifier(n_estimators=5)
    with pytest.raises(ValueError):
        model.fit(X, y)

def test_unfitted_model():
    model = GradientBoostingClassifier()
    X = np.random.randn(10, 2)
    with pytest.raises(ValueError):
        model.predict(X)

def test_compare_with_sklearn():
    # Generate synthetic binary classification data
    np.random.seed(42)
    X = np.random.randn(200, 2)
    y = (X[:, 0] + X[:, 1] + np.random.randn(200) * 0.2 > 0).astype(int)
    
    # Initialize both models with same parameters
    custom_model = GradientBoostingClassifier(
        n_estimators=10,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2
    )
    sklearn_model = SklearnGradientBoostingClassifier(
        n_estimators=10,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        random_state=42
    )
    
    # Train both models
    custom_model.fit(X, y)
    sklearn_model.fit(X, y)
    
    # Make predictions
    custom_predictions = custom_model.predict(X)
    sklearn_predictions = sklearn_model.predict(X)
    
    # Compute accuracies
    custom_accuracy = np.mean(custom_predictions == y)
    sklearn_accuracy = np.mean(sklearn_predictions == y)
    
    # Check that accuracies are within 3% of each other
    assert abs(custom_accuracy - sklearn_accuracy) <= 0.03, \
        f"Custom accuracy ({custom_accuracy:.4f}) differs from sklearn accuracy ({sklearn_accuracy:.4f}) by more than 5%"
