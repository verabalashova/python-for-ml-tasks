import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

def process_data(raw_df: pd.DataFrame):
  # 1. Define input and target columns, and breakdown into train and validation datasets:
  input_cols = list(raw_df.columns)[1:-1]
  target_col = 'Exited'

  X = raw_df[input_cols].copy()
  y = raw_df[target_col].copy()

  train_inputs, val_inputs, train_targets, val_targets = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

  # 2. Define numeric and categorical columns:
  numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
  categorical_cols = train_inputs.select_dtypes(include='object').columns.tolist()

  # note: Id, CustomerId fields have no effect on data analysys, we will remove those.
  # Also `Surname` categorical column that has 748 unique values, we will not encode it.
  numeric_cols = [item for item in numeric_cols if item not in ['id', 'CustomerId']]
  categorical_cols = [item for item in categorical_cols if item != 'Surname']

  # 3. Scale numeric columns with StandartScaler:
  scaler = StandardScaler()
  scaler.fit(train_inputs[numeric_cols])
  train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
  val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])

  # 4. Encode `Geography` and `Gender` categorical columns with OneHotEncoder:
  encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
  encoder.fit(train_inputs[categorical_cols])
  encoded_cols = encoder.get_feature_names_out(categorical_cols)
  train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
  val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])

  model_features = [*numeric_cols, *encoded_cols]

  train_inputs_processed = train_inputs[model_features]
  val_inputs_processed = val_inputs[model_features]

  return {
        'train_X': train_inputs_processed,
        'train_y': train_targets,
        'val_X': val_inputs_processed,
        'val_y': val_targets,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'encoded_cols': encoded_cols,
        'target_col': target_col,
        'scaler': scaler,
        'encoder': encoder
    }

def preprocess_new_data(data_df: pd.DataFrame,
                        numeric_cols,
                        categorical_cols,
                        encoded_cols,
                        scaler: StandardScaler,
                        encoder: OneHotEncoder):
  data_df[numeric_cols] = scaler.transform(data_df[numeric_cols])

  data_df[encoded_cols] = encoder.transform(data_df[categorical_cols])
  training_cols = [*numeric_cols, *encoded_cols]
  transformed_df = data_df[training_cols]

  return transformed_df