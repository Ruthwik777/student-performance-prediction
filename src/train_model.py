from typing import Dict, Any
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def train_models(X_train, y_train, random_state: int = 42) -> Dict[str, Any]:
    log_reg = LogisticRegression(max_iter=1000, random_state=random_state)
    log_reg.fit(X_train, y_train)

    decision_tree = DecisionTreeClassifier(
        max_depth=4,
        random_state=random_state
    )
    decision_tree.fit(X_train, y_train)

    return {
        "Logistic Regression": log_reg,
        "Decision Tree": decision_tree
    }


def save_trained_artifacts(path: str, artifacts: Dict[str, Any]) -> None:
    joblib.dump(artifacts, path)
