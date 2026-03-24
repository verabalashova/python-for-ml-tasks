import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple


# ── Constants ────────────────────────────────────────────────────────────────

EXCLUDED_NUMERIC_COLS = ["id", "CustomerId"]
EXCLUDED_CATEGORICAL_COLS = ["Surname"]
TARGET_COL = "Exited"


# ── Single-responsibility helpers ─────────────────────────────────────────────

def split_features_and_target(
    raw_df: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract input features and the target column from the raw DataFrame.

    The first and last columns are excluded from inputs (assumed to be an index
    and the target respectively).

    Args:
        raw_df: The original, unprocessed DataFrame.
        target_col: Name of the target column.

    Returns:
        A tuple of (X, y) where X is the feature DataFrame and y is the target Series.
    """
    input_cols = list(raw_df.columns)[1:-1]
    X = raw_df[input_cols].copy()
    y = raw_df[target_col].copy()
    return X, y


def split_train_validation(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and target into training and validation sets.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        test_size: Proportion of the dataset to include in the validation split.
        random_state: Seed for reproducibility.

    Returns:
        A tuple of (train_inputs, val_inputs, train_targets, val_targets).
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def identify_feature_columns(
    train_inputs: pd.DataFrame,
    excluded_numeric: list[str] = EXCLUDED_NUMERIC_COLS,
    excluded_categorical: list[str] = EXCLUDED_CATEGORICAL_COLS,
) -> Tuple[list[str], list[str]]:
    """Identify numeric and categorical columns, excluding irrelevant ones.

    Numeric columns such as 'id' and 'CustomerId' are excluded because they
    carry no analytical value. The 'Surname' categorical column is excluded due
    to its high cardinality (748 unique values).

    Args:
        train_inputs: Training feature DataFrame.
        excluded_numeric: Numeric column names to exclude.
        excluded_categorical: Categorical column names to exclude.

    Returns:
        A tuple of (numeric_cols, categorical_cols).
    """
    numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = train_inputs.select_dtypes(include="object").columns.tolist()

    numeric_cols = [c for c in numeric_cols if c not in excluded_numeric]
    categorical_cols = [c for c in categorical_cols if c not in excluded_categorical]

    return numeric_cols, categorical_cols


def fit_and_scale_numeric(
    train_inputs: pd.DataFrame,
    val_inputs: pd.DataFrame,
    numeric_cols: list[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit a StandardScaler on training data and apply it to both splits.

    Args:
        train_inputs: Training feature DataFrame.
        val_inputs: Validation feature DataFrame.
        numeric_cols: List of numeric column names to scale.

    Returns:
        A tuple of (train_inputs, val_inputs, fitted_scaler).
    """
    scaler = StandardScaler()
    scaler.fit(train_inputs[numeric_cols])
    train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
    val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
    return train_inputs, val_inputs, scaler


def fit_and_encode_categorical(
    train_inputs: pd.DataFrame,
    val_inputs: pd.DataFrame,
    categorical_cols: list[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder, list[str]]:
    """Fit a OneHotEncoder on training data and apply it to both splits.

    Args:
        train_inputs: Training feature DataFrame.
        val_inputs: Validation feature DataFrame.
        categorical_cols: List of categorical column names to encode.

    Returns:
        A tuple of (train_inputs, val_inputs, fitted_encoder, encoded_col_names).
    """
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(train_inputs[categorical_cols])
    encoded_cols = encoder.get_feature_names_out(categorical_cols).tolist()
    train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
    val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
    return train_inputs, val_inputs, encoder, encoded_cols


def select_model_features(
    train_inputs: pd.DataFrame,
    val_inputs: pd.DataFrame,
    numeric_cols: list[str],
    encoded_cols: list[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Retain only the final model feature columns in both splits.

    Args:
        train_inputs: Training feature DataFrame (with all intermediate columns).
        val_inputs: Validation feature DataFrame (with all intermediate columns).
        numeric_cols: Scaled numeric column names.
        encoded_cols: One-hot encoded column names.

    Returns:
        A tuple of (train_inputs_processed, val_inputs_processed).
    """
    model_features = [*numeric_cols, *encoded_cols]
    return train_inputs[model_features], val_inputs[model_features]


# ── Public pipeline entry-point ───────────────────────────────────────────────

def preprocess_data(raw_df: pd.DataFrame) -> dict:
    """Run the full preprocessing pipeline on raw training data.

    This is the single function to call instead of the old ``process_data``.
    Internally it delegates to the single-responsibility helpers above.

    Steps performed:
        1. Split raw DataFrame into features (X) and target (y).
        2. Split into train / validation sets.
        3. Identify numeric and categorical columns.
        4. Scale numeric columns with StandardScaler.
        5. Encode categorical columns with OneHotEncoder.
        6. Restrict both splits to final model feature columns.

    Args:
        raw_df: The original, unprocessed DataFrame.

    Returns:
        A dictionary containing:
            - ``train_X`` / ``val_X``: Processed feature DataFrames.
            - ``train_y`` / ``val_y``: Target Series.
            - ``numeric_cols``: Scaled numeric column names.
            - ``categorical_cols``: Source categorical column names.
            - ``encoded_cols``: One-hot encoded column names.
            - ``target_col``: Name of the target column.
            - ``scaler``: Fitted StandardScaler instance.
            - ``encoder``: Fitted OneHotEncoder instance.
    """
    X, y = split_features_and_target(raw_df)
    train_inputs, val_inputs, train_targets, val_targets = split_train_validation(X, y)
    numeric_cols, categorical_cols = identify_feature_columns(train_inputs)

    train_inputs, val_inputs, scaler = fit_and_scale_numeric(
        train_inputs, val_inputs, numeric_cols
    )
    train_inputs, val_inputs, encoder, encoded_cols = fit_and_encode_categorical(
        train_inputs, val_inputs, categorical_cols
    )
    train_inputs_processed, val_inputs_processed = select_model_features(
        train_inputs, val_inputs, numeric_cols, encoded_cols
    )

    return {
        "train_X": train_inputs_processed,
        "train_y": train_targets,
        "val_X": val_inputs_processed,
        "val_y": val_targets,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "encoded_cols": encoded_cols,
        "target_col": TARGET_COL,
        "scaler": scaler,
        "encoder": encoder,
    }


def preprocess_new_data(
    data_df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    encoded_cols: list[str],
    scaler: StandardScaler,
    encoder: OneHotEncoder,
) -> pd.DataFrame:
    """Apply a pre-fitted scaler and encoder to new (unseen) data.

    Uses the scaler and encoder that were fitted during training, so no
    new fitting is performed — only transformations.

    Args:
        data_df: New raw feature DataFrame to transform.
        numeric_cols: Numeric column names to scale.
        categorical_cols: Categorical column names to encode.
        encoded_cols: Expected output column names after one-hot encoding.
        scaler: Fitted StandardScaler from the training pipeline.
        encoder: Fitted OneHotEncoder from the training pipeline.

    Returns:
        A DataFrame containing only the final model feature columns,
        ready for inference.
    """
    data_df[numeric_cols] = scaler.transform(data_df[numeric_cols])
    data_df[encoded_cols] = encoder.transform(data_df[categorical_cols])
    model_features = [*numeric_cols, *encoded_cols]
    return data_df[model_features]
