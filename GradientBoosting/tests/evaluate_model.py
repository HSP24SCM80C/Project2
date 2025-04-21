import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model import GradientBoostingClassifier

def compute_classification_report(y_true, y_pred):
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    classes = np.unique(y_true)
    
    report = {}
    for cls in classes:
        report[cls] = {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0}
    
    # Compute metrics for each class
    for cls in classes:
        true_positives = np.sum((y_true == cls) & (y_pred == cls))
        false_positives = np.sum((y_true != cls) & (y_pred == cls))
        false_negatives = np.sum((y_true == cls) & (y_pred != cls))
        support = np.sum(y_true == cls)
        
        # Precision: TP / (TP + FP)
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        # Recall: TP / (TP + FN)
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        # F1-score: 2 * (precision * recall) / (precision + recall)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        report[cls]['precision'] = precision
        report[cls]['recall'] = recall
        report[cls]['f1-score'] = f1
        report[cls]['support'] = support
    
    # Compute accuracy
    accuracy = np.mean(y_true == y_pred)
    
    # Compute macro averages
    macro_precision = np.mean([report[cls]['precision'] for cls in classes])
    macro_recall = np.mean([report[cls]['recall'] for cls in classes])
    macro_f1 = np.mean([report[cls]['f1-score'] for cls in classes])
    total_support = np.sum([report[cls]['support'] for cls in classes])

    print("\nClassification Report:")
    print("              precision    recall  f1-score   support")
    print()
    for cls in classes:
        print(f"       {cls:>1}    {report[cls]['precision']:.2f}      {report[cls]['recall']:.2f}      {report[cls]['f1-score']:.2f}      {report[cls]['support']}")
    print()
    print(f"    accuracy                        {accuracy:.2f}      {total_support}")
    print(f"   macro avg    {macro_precision:.2f}      {macro_recall:.2f}      {macro_f1:.2f}      {total_support}")
    
    return report, accuracy

def evaluate_model():
    # Load dataset
    try:
        data = np.loadtxt('dataset.csv', delimiter=',', skiprows=1)
    except FileNotFoundError:
        print("Error: 'data/dataset.csv' not found. Please run generate_dataset.py first.")
        return
    
    # Extract features and labels
    X = data[:, :2]
    y = data[:, 2].astype(int)
    
    # Initialize and train the model
    model = GradientBoostingClassifier(
        n_estimators=10,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2
    )
    model.fit(X, y)
    
    # Make predictions
    predictions = model.predict(X)
    
    # Compute and print classification report
    report, accuracy = compute_classification_report(y, predictions)

if __name__ == "__main__":
    evaluate_model()
