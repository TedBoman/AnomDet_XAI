# --- Put this in a file like 'utils.py' ---

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Union
import warnings

def dataframe_to_sequences(
    df: pd.DataFrame,
    sequence_length: int,
    feature_cols: List[str],
    target_col: Optional[str] = None,
    id_col: Optional[str] = None,
    time_col: Optional[str] = None # Optional: For sorting validation
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Converts a pandas DataFrame into windowed sequences (3D NumPy array)
    suitable for time series models. Handles multiple independent series
    if an id_col is provided.

    Args:
        df (pd.DataFrame): Input DataFrame containing the time series data.
                           Must be sorted by time within each group if id_col is used.
        sequence_length (int): The desired length of the output sequences
                               (lookback window size). Must be greater than 0.
        feature_cols (List[str]): A list of column names in `df` to be used as
                                  input features for the sequences.
        target_col (Optional[str]): The column name to be used as the target variable.
                                    If provided, returns (X, y). Defaults to None (return X).
        id_col (Optional[str]): Column identifying independent time series. If None,
                                treats entire DataFrame as one series. Defaults to None.
        time_col (Optional[str]): Column name for time index (for sorting check).

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        - If target_col is None: X (3D NumPy array: n_samples, seq_len, n_features).
        - If target_col is provided: Tuple (X, y) (y is 1D NumPy array).

    Raises:
        ValueError, TypeError: For invalid inputs.
    """
    # --- Input Validation ---
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be greater than 0.")
    if not feature_cols:
        raise ValueError("feature_cols cannot be empty.")
    if not all(col in df.columns for col in feature_cols):
        missing = [col for col in feature_cols if col not in df.columns]
        raise ValueError(f"Feature columns not found in DataFrame: {missing}")
    if target_col and target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
    if id_col and id_col not in df.columns:
        raise ValueError(f"ID column '{id_col}' not found in DataFrame.")
    if time_col and time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found in DataFrame.")

    all_X: List[np.ndarray] = []
    all_y: List[Any] = []
    n_features = len(feature_cols)

    # --- Internal processing function ---
    def process_group(group_df: pd.DataFrame):
        # --- IMPORTANT: Select only feature columns BEFORE getting values ---
        group_features_df = group_df[feature_cols]
        if group_features_df.isnull().values.any():
             warnings.warn(f"NaNs found in feature columns for group. Check input data.", RuntimeWarning)
             # Decide handling: fillna(0)? fillna(method)? skip group? For now, proceed.
             # group_features_df = group_features_df.fillna(0) # Example: Fill NaNs
        group_features = group_features_df.values # Now get NumPy array
        # --- End Change ---

        num_group_samples = len(group_features)
        min_len_required = sequence_length + (1 if target_col else 0)

        if num_group_samples < sequence_length: # Need at least seq_len for one X
            return # Skip this group if too short for even one feature sequence

        group_X: List[np.ndarray] = []
        group_y: List[Any] = []
        group_target = group_df[target_col].values if target_col else None

        # Loop to create sequences
        for i in range(num_group_samples - sequence_length + 1):
            X_seq = group_features[i : i + sequence_length]
            group_X.append(X_seq)

            if target_col:
                target_idx = i + sequence_length
                if target_idx < num_group_samples:
                     group_y.append(group_target[target_idx])
                else: # Reached end, last X has no target
                     group_X.pop() # Remove the last X sequence
                     break

        if group_X:
            all_X.extend(group_X)
            if target_col:
                all_y.extend(group_y)

    # --- Apply processing ---
    if id_col:
        print(f"Processing data grouped by '{id_col}'...")
        grouped = df.groupby(id_col)
        for group_id, group_df in grouped:
             if time_col and not group_df[time_col].is_monotonic_increasing:
                 warnings.warn(f"Time series for ID '{group_id}' not sorted by '{time_col}'. Ensure data is pre-sorted.", UserWarning)
             process_group(group_df)
    else:
        print("Processing data as a single time series...")
        if time_col and not df[time_col].is_monotonic_increasing:
             warnings.warn(f"Time series data not sorted by '{time_col}'. Ensure data is pre-sorted.", UserWarning)
        process_group(df)

    # --- Final Conversion ---
    if not all_X:
        warnings.warn("No sequences generated. Input data might be too short or empty.", UserWarning)
        X_final = np.empty((0, sequence_length, n_features), dtype=np.float32)
        if target_col:
            y_final = np.empty((0,), dtype=np.float32)
            return X_final, y_final
        else: return X_final

    X_final = np.array(all_X)
    if X_final.dtype == 'object':
         try: X_final = X_final.astype(np.float32)
         except ValueError: warnings.warn("Could not convert features to float32.", UserWarning)

    if target_col:
        y_final = np.array(all_y)
        # Try to infer target dtype (handle potential NaNs if filling was used)
        if pd.api.types.is_numeric_dtype(y_final[~np.isnan(y_final)]):
             y_final = y_final.astype(np.float32) # Or float64
        elif pd.api.types.is_bool_dtype(y_final[~np.isnan(y_final)]):
             y_final = y_final.astype(float) # Convert bool to float (0.0/1.0)
        # Keep as object if non-numeric/non-bool and not easily convertible

        print(f"Generated X shape: {X_final.shape}, y shape: {y_final.shape}")
        return X_final, y_final
    else:
        print(f"Generated X shape: {X_final.shape}")
        return X_final

# --- End of utils.py content ---