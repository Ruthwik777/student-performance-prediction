import os

from src.data_preprocessing import load_dataset, split_features_target, preprocess_and_split
from src.train_model import train_models, save_trained_artifacts
from src.evaluate_model import (
    evaluate_classifier,
    plot_confusion_matrices,
    plot_accuracy_comparison,
    plot_feature_importance_decision_tree,
    print_metrics_table
)

def main():
    dataset_path = "dataset/student_data.csv"
    model_path = "models/trained_model.pkl"
    results_dir = "results"

    os.makedirs("models", exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    features = [
        "AttendancePercentage",
        "InternalAssessmentMarks",
        "AssignmentScore",
        "PreviousSemesterMarks",
        "StudyHoursPerWeek"
    ]
    target = "Grade"

    df = load_dataset(dataset_path)
    X, y = split_features_target(df, features, target)
    X_train, X_test, y_train, y_test, artifacts = preprocess_and_split(X, y)

    models = train_models(X_train, y_train)

    predictions = {}
    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        results[name] = evaluate_classifier(y_test, y_pred)

    print_metrics_table(results)

    plot_confusion_matrices(
        y_test, predictions,
        list(artifacts.label_encoder.classes_),
        os.path.join(results_dir, "confusion_matrix.png")
    )

    plot_accuracy_comparison(results, os.path.join(results_dir, "accuracy_comparison.png"))

    plot_feature_importance_decision_tree(
        models["Decision Tree"],
        artifacts.feature_columns,
        os.path.join(results_dir, "feature_importance.png")
    )

    save_trained_artifacts(model_path, {
        "models": models,
        "preprocessing": artifacts
    })

    print("\nProject executed successfully.")

if __name__ == "__main__":
    main()
