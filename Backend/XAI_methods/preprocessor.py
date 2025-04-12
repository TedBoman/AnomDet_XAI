import warnings
import pandas as pd
import numpy as np
from typing import Any, List, Optional, Tuple, Union

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
                                    If provided, the function will also return a `y`
                                    array containing the target value that immediately
                                    follows each sequence. Defaults to None (only return X).
        id_col (Optional[str]): The column name identifying independent time series
                                within the DataFrame. If None, the entire DataFrame
                                is treated as a single sequence. Defaults to None.
        time_col (Optional[str]): The column name for the time index. Used only
                                  for checking if data is sorted when id_col is present.
                                  Defaults to None.

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        - If target_col is None: Returns only X, a 3D NumPy array of shape
          (n_total_samples, sequence_length, n_features).
        - If target_col is provided: Returns a tuple (X, y), where X is the
          3D feature array and y is a 1D NumPy array of target values of shape
          (n_total_samples,).
        - n_total_samples is the total number of valid sequences generated across
          all time series IDs (if applicable).
        - n_features is the number of columns specified in feature_cols.

    Raises:
        ValueError: If input arguments are invalid (e.g., sequence_length <= 0,
                    columns not found, data too short).
        TypeError: If df is not a pandas DataFrame.
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

    # --- Initialization ---
    all_X: List[np.ndarray] = []
    all_y: List[Any] = [] # Use Any for potential different target types initially
    n_features = len(feature_cols)

    # --- Define processing logic for a single series (or group) ---
    def process_group(group_df: pd.DataFrame):
        group_features = group_df[feature_cols].values
        num_group_samples = len(group_features)

        # Check if group is long enough to create at least one sequence
        min_len_required = sequence_length + (1 if target_col else 0)
        if num_group_samples < min_len_required:
            return # Skip this group if too short

        group_X: List[np.ndarray] = []
        group_y: List[Any] = []

        group_target = group_df[target_col].values if target_col else None

        # Generate sequences for this group
        # The loop range ensures we don't go out of bounds for X or y
        for i in range(num_group_samples - sequence_length + (0 if target_col else 1)):
            # Check if we can still form a full sequence of length `sequence_length`
            if i + sequence_length > num_group_samples:
                 break # Should not happen with correct loop range, but safety check

            X_seq = group_features[i : i + sequence_length]
            group_X.append(X_seq)

            # Get target if requested (value *after* the sequence ends)
            if target_col:
                target_idx = i + sequence_length
                # Ensure target index is valid
                if target_idx < num_group_samples:
                     group_y.append(group_target[target_idx])
                else:
                     # This case means we created an X sequence but there's no target for it
                     # We should remove the last X sequence added
                     group_X.pop()
                     break # Stop processing for this group

        # Append results from this group to the main lists
        if group_X:
            all_X.extend(group_X)
            if target_col:
                all_y.extend(group_y)

    # --- Apply processing ---
    if id_col:
        print(f"Processing data grouped by '{id_col}'...")
        grouped = df.groupby(id_col)
        for group_id, group_df in grouped:
             # Optional: Validate sorting within group if time_col provided
             if time_col:
                 if not group_df[time_col].is_monotonic_increasing:
                     warnings.warn(
                         f"Time series for ID '{group_id}' is not sorted by '{time_col}'. "
                         "Sequences might be incorrect. Ensure data is pre-sorted.",
                         UserWarning
                     )
             process_group(group_df)
    else:
        print("Processing data as a single time series...")
        # Optional: Validate sorting if time_col provided
        if time_col:
            if not df[time_col].is_monotonic_increasing:
                 warnings.warn(
                     f"Time series data is not sorted by '{time_col}'. "
                     "Sequences might be incorrect. Ensure data is pre-sorted.",
                     UserWarning
                 )
        process_group(df)

    # --- Final Conversion and Return ---
    if not all_X:
        warnings.warn("No sequences generated. Input data might be too short or empty.", UserWarning)
        # Return empty arrays with correct dimensions
        X_final = np.empty((0, sequence_length, n_features), dtype=np.float32) # Use float32 or infer
        if target_col:
            y_final = np.empty((0,), dtype=np.float32) # Or infer dtype from all_y if non-empty
            return X_final, y_final
        else:
            return X_final

    # Convert lists to NumPy arrays
    X_final = np.array(all_X)
    # Infer dtype, default to float32 if possible, otherwise use original
    if X_final.dtype == 'object': # If features had mixed types
         try:
            X_final = X_final.astype(np.float32)
         except ValueError:
            warnings.warn("Could not convert features to float32, retaining object dtype.", UserWarning)


    if target_col:
        y_final = np.array(all_y)
         # Try to infer target dtype
        if y_final.dtype == 'object':
             try:
                 # Attempt numeric conversion first
                 y_final = y_final.astype(np.float32)
             except ValueError:
                 # Keep as object if conversion fails (e.g., string targets)
                  warnings.warn("Could not convert target to float32, retaining object dtype.", UserWarning)
        print(f"Generated X shape: {X_final.shape}, y shape: {y_final.shape}")
        return X_final, y_final
    else:
        print(f"Generated X shape: {X_final.shape}")
        return X_final