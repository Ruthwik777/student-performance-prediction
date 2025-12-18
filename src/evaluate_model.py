from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def evaluate_classifier(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0)
    }


def plot_confusion_matrices(y_true, predictions, class_names, save_path):
    fig, axes = plt.subplots(1, len(predictions), figsize=(6 * len(predictions), 5))
    if len(predictions) == 1:
        axes = [axes]

    for ax, (name, y_pred) in zip(axes, predictions.items()):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_accuracy_comparison(results, save_path):
    models = list(results.keys())
    accuracies = [results[m]["accuracy"] for m in models]

    plt.figure(figsize=(6, 4))
    plt.bar(models, accuracies)
    plt.ylim(0, 1)
    plt.title("Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_feature_importance_decision_tree(model, feature_names, save_path):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(7, 4))
    plt.bar([feature_names[i] for i in indices], importances[indices])
    plt.xticks(rotation=20)
    plt.title("Decision Tree Feature Importance")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def print_metrics_table(results):
    print("\nMODEL EVALUATION RESULTS")
    print("-" * 50)
    for model, m in results.items():
        print(f"\n{model}")
        for k, v in m.items():
            print(f"{k.capitalize():10}: {v:.4f}")
    print("-" * 50)
