import pandas as pd
import numpy as np
from typing import Any, List, Optional, Tuple, Union
import warnings

def select_explanation_indices(
    df: pd.DataFrame, 
    strategy: str, 
    n: int, 
    label_col: Optional[str] = None
) -> np.ndarray:
    """
    Selects indices from a DataFrame based on a specified sampling strategy.

    Args:
        df (pd.DataFrame): The full dataset.
        strategy (str): The sampling strategy. Valid options:
            'first_n', 'random', 'random_anomalies', 'first_n_anomalies',
            'last_n_anomalies', 'half_n_half'.
        n (int): The desired number of indices.
        label_col (Optional[str]): The name of the column containing ground truth
                                   anomaly labels (0=normal, 1=anomaly). 
                                   Required for anomaly-based strategies.

    Returns:
        np.ndarray: An array of integer indices selected from the DataFrame's index.
                    Returns fewer than n indices if insufficient data is available
                    for the chosen strategy (e.g., fewer than n anomalies).
                    
    Raises:
        ValueError: If an invalid strategy is provided, or if label_col is required
                    but not provided or not found in the DataFrame.
        TypeError: If df is not a Pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a Pandas DataFrame.")
        
    n_total = len(df)
    if n <= 0:
        warnings.warn("Number of samples 'n' must be positive. Returning empty array.")
        return np.array([], dtype=int)
        
    # Ensure n does not exceed total available samples
    n = min(n, n_total) 
    if n == 0: # Handle case where df might be empty
         return np.array([], dtype=int)

    # Use DataFrame's integer-based index for selection
    possible_indices = df.index.to_numpy() 

    selected_indices = np.array([], dtype=int)

    # --- Strategy Implementation ---
    if strategy == 'first_n':
        selected_indices = possible_indices[:n]
        
    elif strategy == 'random':
        selected_indices = np.random.choice(possible_indices, size=n, replace=False)
        
    elif strategy in ['random_anomalies', 'first_n_anomalies', 'last_n_anomalies', 'half_n_half']:
        if label_col is None:
            raise ValueError(f"Strategy '{strategy}' requires 'label_col' to be provided.")
        if label_col not in df.columns:
             raise ValueError(f"Label column '{label_col}' not found in DataFrame.")
             
        try:
            # Ensure labels are treated as integers/booleans
            labels = df[label_col].astype(bool) 
        except Exception as e:
            raise ValueError(f"Could not interpret label column '{label_col}' as boolean/numeric (0/1). Error: {e}") from e
            
        anomaly_indices = possible_indices[labels]
        normal_indices = possible_indices[~labels]
        n_anomalies_avail = len(anomaly_indices)
        n_normals_avail = len(normal_indices)

        if strategy == 'random_anomalies':
            n_to_select = min(n, n_anomalies_avail)
            if n_to_select < n:
                warnings.warn(f"Requested {n} anomalies, but only {n_anomalies_avail} available. Selecting {n_to_select}.")
            if n_anomalies_avail > 0:
                selected_indices = np.random.choice(anomaly_indices, size=n_to_select, replace=False)
            else:
                 warnings.warn("No anomalies found in the dataset.")

        elif strategy == 'first_n_anomalies':
            n_to_select = min(n, n_anomalies_avail)
            if n_to_select < n:
                warnings.warn(f"Requested {n} first anomalies, but only {n_anomalies_avail} available. Selecting {n_to_select}.")
            selected_indices = anomaly_indices[:n_to_select]

        elif strategy == 'last_n_anomalies':
            n_to_select = min(n, n_anomalies_avail)
            if n_to_select < n:
                warnings.warn(f"Requested {n} last anomalies, but only {n_anomalies_avail} available. Selecting {n_to_select}.")
            selected_indices = anomaly_indices[-n_to_select:]
            
        elif strategy == 'half_n_half':
            n_anom_target = n // 2
            n_norm_target = n - n_anom_target
            
            n_anom_select = min(n_anom_target, n_anomalies_avail)
            n_norm_select = min(n_norm_target, n_normals_avail)

            if n_anom_select < n_anom_target:
                warnings.warn(f"Requested {n_anom_target} anomalies for half-n-half, but only {n_anomalies_avail} available. Selecting {n_anom_select}.")
            if n_norm_select < n_norm_target:
                 warnings.warn(f"Requested {n_norm_target} normals for half-n-half, but only {n_normals_avail} available. Selecting {n_norm_select}.")

            anom_indices_selected = np.array([], dtype=int)
            norm_indices_selected = np.array([], dtype=int)
                 
            if n_anomalies_avail > 0 and n_anom_select > 0:
                 anom_indices_selected = np.random.choice(anomaly_indices, size=n_anom_select, replace=False)
            if n_normals_avail > 0 and n_norm_select > 0:
                 norm_indices_selected = np.random.choice(normal_indices, size=n_norm_select, replace=False)
                 
            selected_indices = np.concatenate((anom_indices_selected, norm_indices_selected))
            # Optional: Shuffle the combined indices if order doesn't matter
            # np.random.shuffle(selected_indices) 

    # --- Add other strategies like 'boundary' or 'errors' here if needed ---
    # Example for 'boundary' (needs probabilities):
    # elif strategy == 'boundary':
    #     if pred_proba_col is None or pred_proba_col not in df.columns:
    #         raise ValueError("Strategy 'boundary' requires 'pred_proba_col'.")
    #     diff_from_half = np.abs(df[pred_proba_col] - 0.5)
    #     sorted_indices = diff_from_half.sort_values().index.to_numpy()
    #     selected_indices = sorted_indices[:n]

    else:
        raise ValueError(f"Unknown sampling strategy: '{strategy}'. Valid options are: "
                         f"'first_n', 'random', 'random_anomalies', 'first_n_anomalies', "
                         f"'last_n_anomalies', 'half_n_half'.") # Add others as implemented

    # Ensure unique indices if concatenation happened (e.g., half_n_half)
    # Although sampling without replacement should prevent duplicates here.
    # selected_indices = np.unique(selected_indices) 
    
    print(f"Selected {len(selected_indices)} indices using strategy '{strategy}'.")
    return selected_indices.astype(int) # Ensure integer type

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