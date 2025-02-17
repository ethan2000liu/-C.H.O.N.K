import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def load_model_and_data():
    # Load the best model
    model = tf.keras.models.load_model('models/best_model.keras')
    
    # Load test data
    X_test = np.load('processed_data/X_test.npy')
    y_test = np.load('processed_data/y_test.npy')
    
    return model, X_test, y_test

def plot_confusion_matrix(y_true, y_pred):
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred.round())
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('evaluation/confusion_matrix.png')
    plt.close()

def plot_roc_curve(y_true, y_pred_proba):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('evaluation/roc_curve.png')
    plt.close()

def main():
    # Create evaluation directory
    import os
    os.makedirs('evaluation', exist_ok=True)
    
    # Load model and data
    model, X_test, y_test = load_model_and_data()
    
    # Get predictions
    y_pred_proba = model.predict(X_test)
    y_pred = y_pred_proba.round()
    
    # Generate classification report
    report = classification_report(y_test, y_pred, target_names=['Normal', 'Chonky'])
    print("\nClassification Report:")
    print(report)
    
    # Save report to file
    with open('evaluation/classification_report.txt', 'w') as f:
        f.write(report)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred)
    
    # Plot ROC curve
    plot_roc_curve(y_test, y_pred_proba)
    
    print("\nEvaluation completed! Check the 'evaluation' folder for detailed results.")

if __name__ == "__main__":
    main() 