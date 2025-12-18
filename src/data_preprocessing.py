from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


@dataclass
class PreprocessingArtifacts:
    imputer: SimpleImputer
    scaler: StandardScaler
    label_encoder: LabelEncoder
    feature_columns: List[str]
    target_column: str


def load_dataset(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def split_features_target(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str
) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[feature_columns].copy()
    y = df[target_column].copy()
    return X, y


def preprocess_and_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
):
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y.astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded
    )

    artifacts = PreprocessingArtifacts(
        imputer=imputer,
        scaler=scaler,
        label_encoder=label_encoder,
        feature_columns=list(X.columns),
        target_column=y.name
    )

    return X_train, X_test, y_train, y_test, artifacts
