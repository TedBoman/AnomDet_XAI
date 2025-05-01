import sys
import warnings
import pandas as pd
import numpy as np
import os
import time
import threading
from socket import socket
from timescaledb_api import TimescaleDBAPI
from datetime import datetime, timezone
from typing import Any, Union
import multiprocessing as mp
import threading

# Third-Party
from pathlib import Path

import pandas as pd
from typing import Union, List, Optional, Dict

# Custom
from Simulator.DBAPI.type_classes import Job
from Simulator.DBAPI.type_classes import AnomalySetting
from Simulator.SimulatorEngine import SimulatorEngine as se
from ML_models.get_model import get_model

# --- XAI ---
from ML_models.model_wrapper import ModelWrapperForXAI
import XAI_methods.preprocessor as pre
from XAI_methods.timeseriesExplainer import TimeSeriesExplainer
import shap
import matplotlib.pyplot as plt
from XAI_methods import xai_visualizations as x
import utils as ut

MODEL_DIRECTORY = "./ML_models"
INJECTION_METHOD_DIRECTORY = "./Simulator/AnomalyInjector/InjectionMethods"
XAI_METHOD_DIRECTORY = "/XAI_methods/methods"
DATASET_DIRECTORY = "./Datasets"

def get_anomaly_rows(
    data: pd.DataFrame,
    label_column: str = 'label', # Common name for the label column
    anomaly_value: Any = 1       # The value indicating an anomaly (often 1 or True)
) -> pd.DataFrame:
    """
    Filters a pandas DataFrame to return only the rows marked as anomalies
    based on a specific label column and value.

    Args:
        data (pd.DataFrame): The input DataFrame containing the data and labels.
        label_column (str): The name of the column containing the anomaly labels.
                            Defaults to 'label'. Common alternatives might be
                            'is_anomaly', 'anomaly', 'target'.
        anomaly_value (Any): The value within the `label_column` that signifies
                             an anomaly. Defaults to 1. This could also be True,
                             'anomaly', etc., depending on your dataset.

    Returns:
        pd.DataFrame: A new DataFrame containing only the rows from the input `data`
                      where the value in the `label_column` equals `anomaly_value`.
                      Preserves the original index and columns. Returns an empty
                      DataFrame with the same columns if no anomalies are found
                      or if the input DataFrame is empty.

    Raises:
        TypeError: If 'data' is not a pandas DataFrame.
        ValueError: If 'label_column' is not found in the DataFrame's columns.
    """
    # 1. Input Validation
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input 'data' must be a pandas DataFrame.")

    if label_column not in data.columns:
        raise ValueError(
            f"Label column '{label_column}' not found in DataFrame columns. "
            f"Available columns: {data.columns.tolist()}"
        )

    if data.empty:
        print("Input DataFrame is empty. Returning an empty DataFrame.")
        # Return an empty frame structure matching the input
        return pd.DataFrame(columns=data.columns).astype(data.dtypes)

    # 2. Filtering Logic
    try:
        # Create a boolean mask where the condition is true
        anomaly_mask = (data[label_column] == anomaly_value)

        # Use the mask to select rows. Use .copy() to avoid potential
        # SettingWithCopyWarning if the returned DataFrame is modified later.
        anomaly_rows = data.loc[anomaly_mask].copy()

        print(f"Found {len(anomaly_rows)} rows where '{label_column}' == {anomaly_value}.")

    except Exception as e:
        # Catch potential errors during comparison or indexing
        print(f"An error occurred during filtering: {e}")
        # Return an empty DataFrame with original structure on error
        return pd.DataFrame(columns=data.columns).astype(data.dtypes)

    # 3. Return Filtered DataFrame
    return anomaly_rows

def get_balanced_anomaly_sample(
    data: pd.DataFrame,
    total_rows: int, # Parameter for the desired total number of rows
    label_column: str = 'label',
    anomaly_value: Any = 1,
    random_state: Optional[int] = None # Still useful for internal consistency if needed, but sampling is not random
) -> pd.DataFrame:
    """
    Filters a pandas DataFrame to return a new DataFrame containing a sample
    of anomaly and non-anomaly rows, aiming for a balanced distribution,
    up to a specified total number of rows, while preserving temporal order
    by sampling from the end of each sequence.

    The function samples the most recent `N_anomaly` anomalies and the most
    recent `N_non_anomaly` non-anomalies, where N_anomaly + N_non_anomaly
    equals the minimum of:
    1. The requested `total_rows`.
    2. The total number of available rows in the input `data`.
    3. Twice the count of the less frequent class (to maintain balance if possible).

    The function prioritizes achieving the `total_rows` while keeping the
    anomaly/non-anomaly split as close to 50/50 as possible given the constraints.
    The output DataFrame will contain the sampled anomalies followed by the
    sampled non-anomalies, maintaining their original relative order within
    each group.

    Args:
        data (pd.DataFrame): The input DataFrame containing the data and labels.
                             Assumes data is already sorted by time/index.
        total_rows (int): The desired total number of rows in the output DataFrame.
                          Must be a non-negative integer.
        label_column (str): The name of the column containing the anomaly labels.
                            Defaults to 'label'.
        anomaly_value (Any): The value within the `label_column` that signifies
                             an anomaly. Defaults to 1.
        random_state (Optional[int]): Seed for any potential internal random
                                      operations (though primary sampling is
                                      now temporal).

    Returns:
        pd.DataFrame: A new DataFrame containing the sampled rows, preserving
                      temporal order within each class and concatenating them.
                      Preserves original columns. Returns an empty DataFrame
                      with the same columns if the input is empty, total_rows
                      is 0, or no instances of a required class are available.

    Raises:
        TypeError: If 'data' is not a pandas DataFrame or total_rows is not an int.
        ValueError: If 'label_column' is not found or total_rows is negative.
    """
    # 1. Input Validation
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input 'data' must be a pandas DataFrame.")

    if not isinstance(total_rows, int):
        raise TypeError("Input 'total_rows' must be an integer.")

    if total_rows < 0:
        raise ValueError("Input 'total_rows' must be a non-negative integer.")

    if label_column not in data.columns:
        raise ValueError(
            f"Label column '{label_column}' not found in DataFrame columns. "
            f"Available columns: {data.columns.tolist()}"
        )

    if data.empty or total_rows == 0:
        print("Input DataFrame is empty or total_rows is 0. Returning an empty DataFrame.")
        return pd.DataFrame(columns=data.columns).astype(data.dtypes)

    # Note: random_state is less critical now as sampling is temporal,
    # but kept for potential future use or internal pandas operations.
    np.random.seed(random_state)

    # 2. Separate Anomalies and Non-Anomalies (preserving original order)
    try:
        anomaly_mask = (data[label_column] == anomaly_value)
        # Use .loc and keep original order
        anomaly_df = data.loc[anomaly_mask].copy()
        non_anomaly_df = data.loc[~anomaly_mask].copy() # Use inverse mask

        n_anomalies = len(anomaly_df)
        n_non_anomalies = len(non_anomaly_df)

        print(f"Found {n_anomalies} anomaly rows and {n_non_anomalies} non-anomaly rows.")

    except Exception as e:
        print(f"An error occurred during data separation: {e}")
        return pd.DataFrame(columns=data.columns).astype(data.dtypes)

    # Check if any class is empty before proceeding
    if n_anomalies == 0 and n_non_anomalies == 0:
         print("Input DataFrame contains no rows. Returning empty DataFrame.")
         return pd.DataFrame(columns=data.columns).astype(data.dtypes)
    elif n_anomalies == 0:
         print("No anomaly instances found. Returning up to total_rows from the end of non-anomalies.")
         # If no anomalies, return up to total_rows from the end of non-anomalies
         num_non_anomalies_to_sample = min(total_rows, n_non_anomalies)
         # Use .tail() to get the last rows
         sampled_df = non_anomaly_df.tail(n=num_non_anomalies_to_sample)
         # Reset index but DO NOT shuffle
         return sampled_df.reset_index(drop=True)
    elif n_non_anomalies == 0:
         print("No non-anomaly instances found. Returning up to total_rows from the end of anomalies.")
         # If no non-anomalies, return up to total_rows from the end of anomalies
         num_anomalies_to_sample = min(total_rows, n_anomalies)
         # Use .tail() to get the last rows
         sampled_df = anomaly_df.tail(n=num_anomalies_to_sample)
         # Reset index but DO NOT shuffle
         return sampled_df.reset_index(drop=True)


    # 3. Determine Actual Number of Rows to Return
    actual_total_to_return = min(total_rows, n_anomalies + n_non_anomalies)
    print(f"Aiming to return {actual_total_to_return} rows (min of requested {total_rows} and available {n_anomalies + n_non_anomalies}).")


    # 4. Determine Number of Samples from Each Class (Aiming for Balance)
    # Calculate target counts for a 50/50 split
    target_anomalies = actual_total_to_return // 2
    target_non_anomalies = actual_total_to_return - target_anomalies # Handles odd total

    # Calculate initial sample counts, limited by available data
    num_anomalies_to_sample = min(target_anomalies, n_anomalies)
    num_non_anomalies_to_sample = min(target_non_anomalies, n_non_anomalies)

    # Calculate how many more samples are needed to reach actual_total_to_return
    current_total = num_anomalies_to_sample + num_non_anomalies_to_sample
    remaining_needed = actual_total_to_return - current_total

    # Distribute the remaining needed samples to the class with more available capacity
    # This ensures we hit actual_total_to_return while staying within class limits
    if remaining_needed > 0:
        anomaly_capacity_left = n_anomalies - num_anomalies_to_sample
        non_anomaly_capacity_left = n_non_anomalies - num_non_anomalies_to_sample

        # Prioritize adding to the class that has more available instances remaining
        if anomaly_capacity_left > non_anomaly_capacity_left:
            add_anomalies = min(remaining_needed, anomaly_capacity_left)
            num_anomalies_to_sample += add_anomalies
            remaining_needed -= add_anomalies

        if remaining_needed > 0:
            add_non_anomalies = min(remaining_needed, non_anomaly_capacity_left)
            num_non_anomalies_to_sample += add_non_anomalies
            # remaining_needed should now be 0

    print(f"Sampling {num_anomalies_to_sample} anomalies and {num_non_anomalies_to_sample} non-anomalies from the end of each sequence.")


    # 5. Sample from the End of Each Group (Temporal Sampling)
    try:
        # Use .tail() to get the last 'n' rows, preserving their order
        sampled_anomalies = anomaly_df.tail(n=num_anomalies_to_sample)
        sampled_non_anomalies = non_anomaly_df.tail(n=num_non_anomalies_to_sample)

    except Exception as e:
        print(f"An unexpected error occurred during temporal sampling: {e}")
        return pd.DataFrame(columns=data.columns).astype(data.dtypes)


    # 6. Combine (DO NOT SHUFFLE)
    try:
        # Concatenate the two sampled DataFrames. The order will be
        # sampled_anomalies followed by sampled_non_anomalies.
        balanced_sample_df = pd.concat([sampled_anomalies, sampled_non_anomalies])

        # Reset index to get a clean 0-based index for the new combined DataFrame,
        # but explicitly do not shuffle.
        balanced_sample_df = balanced_sample_df.reset_index(drop=True)

        print(f"Created temporal sample with {len(balanced_sample_df)} rows.")
        if not balanced_sample_df.empty:
            print("Value counts in sample:\n", balanced_sample_df[label_column].value_counts())

    except Exception as e:
        print(f"An error occurred during concatenation or index reset: {e}")
        return pd.DataFrame(columns=data.columns).astype(data.dtypes)


    # 7. Return Sampled DataFrame
    return balanced_sample_df

def split_data(data):
    """Split the dataseries into 2 data series in a ratio of """

    total_rows = len(data)

    # Calculate split indices

    train_end = int(total_rows * 0.70) # 70% for training

    # Split the data
    training_data = data.iloc[:train_end]
    testing_data = data.iloc[train_end:] # Remaining 30% is testing

    return training_data, testing_data

def map_to_timestamp(time):
    return time.timestamp()

def map_to_time(time):
    return datetime.fromtimestamp(time, tz=timezone.utc)

def evaluate_classification(df: pd.DataFrame) -> dict:
    # Ensure the DataFrame has the necessary columns
    if "is_anomaly" not in df.columns:
        raise ValueError("DataFrame must contain an 'is_anomaly' column.")
    if "label" not in df.columns:
        raise ValueError("DataFrame must contain a 'label' column.")

    # Convert boolean columns to integers for easier comparison (True=1, False=0)
    df["predicted"] = df["is_anomaly"].astype(int)
    df["actual"] = df["label"].astype(int)

    # Calculate evaluation metrics
    correct_anomalies = df[(df["predicted"] == 1) & (df["actual"] == 1)].shape[0]
    correct_non_anomalies = df[(df["predicted"] == 0) & (df["actual"] == 0)].shape[0]
    false_positives = df[(df["predicted"] == 1) & (df["actual"] == 0)].shape[0]
    false_negatives = df[(df["predicted"] == 0) & (df["actual"] == 1)].shape[0]

    total_predictions = len(df)
    accuracy = (correct_anomalies + correct_non_anomalies) / total_predictions if total_predictions > 0 else 0

    return {
        "correct_anomalies": correct_anomalies,
        "correct_non_anomalies": correct_non_anomalies,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "total_predictions": total_predictions,
        "accuracy": accuracy,
    }

# Starts processing of dataset in one batch
def run_batch(
    db_conn_params: Dict,
    model: str,                          # Model name string (e.g., 'lstm', 'svm')
    path: str,                           # Path to dataset file
    name: str,                           # Job name (e.g., "job_batch_myjob")
    inj_params: Optional[List[Dict[str, Any]]] = None, 
    debug: bool = False,
    label_column: Optional[str] = None, 
    xai_params: Optional[Dict[str, Any]] = None,
    model_params: Optional[Dict[str, Any]] = None,
) -> None:    
    print("Starting Batch-job!")
    sys.stdout.flush()
    
    start_time = time.perf_counter()

    if inj_params is not None:
        anomaly_settings = []  # Create a list to hold AnomalySetting objects
        for params in inj_params:  # Iterate over the list of anomaly dictionaries
            anomaly = AnomalySetting(
                params.get("anomaly_type", None),
                int(params.get("timestamp", None)),
                int(params.get("magnitude", None)),
                int(params.get("percentage", None)),
                params.get("columns", None),
                params.get("duration", None)
            )
            anomaly_settings.append(anomaly)  # Add the AnomalySetting object to the list

        batch_job = Job(filepath=path, anomaly_settings=anomaly_settings, simulation_type="batch", speedup=None, table_name=name, debug=debug)
        
    else:
        batch_job = Job(filepath=path, simulation_type="batch", anomaly_settings=None, speedup=None, table_name=name, debug=debug)
    sim_engine = se()
    result = sim_engine.main(db_conn_params=db_conn_params, job=batch_job, timestamp_col_name=None, label_col_name=label_column) # TODO: Change the None to timestamp column name when added
    end_time = time.perf_counter()
    print(f"Batch import took {end_time-start_time}s")

    if result == 1:
        api = TimescaleDBAPI(db_conn_params)
        print(f"Reading data for table/job: {name}")
        try:
            df = api.read_data(datetime.fromtimestamp(0), name)
            if df.empty: raise ValueError("DataFrame read from DB is empty.")
        except Exception as e:
            print(f"Error reading data from TimescaleDB for '{name}': {e}")
            return # Exit if no data
        
        # --- Data Splitting ---
        try:
             training_data, testing_data = split_data(df)
             if training_data.empty or testing_data.empty:
                 warnings.warn("Training or testing data split resulted in empty DataFrame.", RuntimeWarning)
                 # Handle fallback? Maybe use whole df for training if test is empty? Requires care.
        except Exception as e:
             print(f"Error during data splitting: {e}")
             return # Cannot proceed without data splits

        # --- Feature Column Definition ---
        # Define default feature columns (adjust slicing as needed!)
        # Example: Exclude first ('timestamp'?) and last three ('label', 'inj_anomaly', 'is_anomaly'?)
        try:
            if model == 'XGBoost' or model == 'decision_tree': # Handle specific model cases if needed
                feature_columns = df.columns[1:-3].tolist()
                training_columns = df.columns[1:-2].tolist()
            else:
                # Ensure indices are valid for the dataframe shape
                if df.shape[1] > 4: # Need at least 5 columns for [1:-3]
                    feature_columns = df.columns[1:-3].tolist()
                    training_columns = df.columns[1:-3].tolist()
                elif df.shape[1] > 1: # If fewer columns, take all except first
                    feature_columns = df.columns[1:].tolist()
                    training_columns = df.columns[1:].tolist()
                    warnings.warn("DataFrame has few columns, using all except the first as features.", RuntimeWarning)
                else:
                    raise ValueError("DataFrame has too few columns to determine features.")
        except Exception as e:
             print(f"Error determining feature columns: {e}")
             return

        if not feature_columns:
             print("Error: No feature columns identified. Stopping.")
             return
        
        actual_label_col = 'label' # Use provided label or default
        anomaly_features = feature_columns + [actual_label_col]
        print(f"Selected Feature Columns: {feature_columns}")
        print(f"Selected Anomaly Columns with label: {anomaly_features}")
        print(f"Selected Trainigng Columns: {training_columns}")

        # Select features (handle potential missing columns defensively)
        training_features_df = training_data[[col for col in training_columns if col in training_data.columns]]
        training_df_with_labels = training_data
        testing_features_df = testing_data[[col for col in feature_columns if col in testing_data.columns]]
        testing_df_with_labels = testing_data
        all_features_df = df[[col for col in feature_columns if col in df.columns]]
        all_features_df_with_labels = df

        #balanced_df = get_balanced_anomaly_sample(all_features_df_with_labels, total_rows=len(df), label_column='label', anomaly_value=1, random_state=42)

        # Define which base features are continuous (example: assume all for now)
        continuous_features_list = feature_columns # Adjust if you have categorical base features

        print("Training features shape:", training_features_df.shape)
        print("Testing features shape:", testing_features_df.shape)
        print("All features shape:", all_features_df.shape)
        if training_features_df.empty:
            print("Error: Training features DataFrame is empty after column selection.")
            return
        # --- End Feature Selection ---

        # --- Anomaly Row Extraction (using label_column parameter) ---
        anomaly_feature_df = pd.DataFrame(columns=anomaly_features) # Initialize empty DF with correct columns
        print(f"Attempting to use '{actual_label_col}' as the label column for anomaly extraction.")
        if actual_label_col in df.columns:
            try:
                 anomaly_rows = get_anomaly_rows(df, label_column=actual_label_col, anomaly_value=1)
                 if not anomaly_rows.empty:
                      # Select only the defined feature columns if they exist in anomaly rows
                      cols_in_anomaly = [col for col in feature_columns if col in anomaly_rows.columns]
                      if cols_in_anomaly:
                           anomaly_feature_df = anomaly_rows[cols_in_anomaly]
                      else: print("Warning: Feature columns not present in found anomaly rows.")
                 print(f"Found {len(anomaly_feature_df)} anomaly rows with valid features.")
            except Exception as e:
                 print(f"Error getting/processing anomaly rows using label '{actual_label_col}': {e}")
        else:
             print(f"Warning: Specified label column '{actual_label_col}' not found in DataFrame. Cannot extract specific anomaly rows.")
        # --- End Anomaly Row Extraction ---

        # --- Model Training ---
        try:
            # Prepare parameters for the model, ensuring None is handled
            effective_model_params = model_params or {}
            print(f"Attempting to get model '{model}' with parameters: {effective_model_params}")

            # Pass model_params during instantiation via get_model
            # This assumes get_model and the underlying model classes
            # accept these parameters as keyword arguments.
            model_instance = get_model(model, **effective_model_params)

            print(f"Training model type: {type(model_instance).__name__}")
            start_time = time.perf_counter()

            # Train on TRAINING features DataFrame
            # If the run method *also* takes specific training-time params (like epochs),
            # those might still be passed here or could be part of model_params as well.
            # Assuming basic run just takes data for now based on original code.
            model_instance.run(training_features_df)

            end_time = time.perf_counter()
            print(f"Training took {end_time-start_time:.2f}s")
        except Exception as train_err:
            print(f"ERROR during model retrieval or training: {train_err}")
            import traceback
            traceback.print_exc() # Print full traceback for better debugging
            return # Stop if model cannot be trained

        # --- Sequence Length Determination ---
        sequence_length = getattr(model_instance, 'sequence_length', None)
        if not isinstance(sequence_length, int) or sequence_length <= 0:
             # If model isn't sequential or attr missing, set default for XAI framework IF XAI is requested
             if xai_params:
                 default_xai_seq_len = 10 # Default sequence length for XAI framework if model is non-sequential
                 warnings.warn(f"Model doesn't provide positive integer 'sequence_length'. Using default={default_xai_seq_len} for XAI data preparation.", RuntimeWarning)
                 sequence_length = default_xai_seq_len # Use default ONLY for XAI prep
             else:
                 sequence_length = 1 # Or None if XAI isn't running anyway
                 print("Model does not appear sequential or sequence_length missing. Proceeding without sequence assumption for padding.")
        else:
            print(f"Determined sequence_length from model: {sequence_length}")

        # --- Create Wrapper (Conditional - only if XAI is running) ---
        model_wrapper = None
        if xai_params and isinstance(xai_params, list): # Check if XAI is requested
            interpretation = 'higher_is_anomaly' # Default
            model_type_str = model.lower() # Use the input string for type check
            if 'svm' in model_type_str: interpretation = 'lower_is_anomaly'
            elif 'lstm' in model_type_str: interpretation = 'higher_is_anomaly'
            elif 'xgboost' in model_type_str: interpretation = 'higher_is_anomaly'
            elif 'decision_tree' in model_type_str: interpretation = 'higher_is_anomaly'
            # Add other model types here
            else: warnings.warn(f"Unknown model type '{model}' for score interpretation. Assuming higher score is anomaly.", RuntimeWarning)

            if sequence_length is None: # Should have been set above if XAI is running
                 print("ERROR: sequence_length required for XAI wrapper but is None. Skipping XAI.")
            else:
                try:
                    print(f"Wrapping trained model instance ({type(model_instance).__name__}) for XAI...")
                    # Choose the correct wrapper based on model's expected input
                    model_wrapper = ModelWrapperForXAI(
                        actual_model_instance=model_instance,
                        feature_names=feature_columns,
                        score_interpretation=interpretation
                    )
                    # Associate the XAI sequence length with the wrapper if needed by TimeSeriesExplainer checks

                    print(f"Model wrapped successfully. Score interpretation: '{interpretation}'.")
                except Exception as e:
                    print(f"Error wrapping model: {e}. Skipping XAI.")
                    model_wrapper = None
        # --- End Wrapper ---

        # --- Anomaly Detection ---
        # Process timestamps if needed AFTER getting features (assuming map_to_timestamp not needed for model)
        df_eval = df.copy() # Create copy for timestamp manipulation if needed for evaluation/update
        df_eval["timestamp"] = df_eval["timestamp"].apply(map_to_timestamp)
        df_eval["timestamp"] = df_eval["timestamp"].astype(float)

        start_time = time.perf_counter()
        res = model_instance.detect(all_features_df) # Detect using selected feature columns
        end_time = time.perf_counter()
        print(f"Anomaly detection took {end_time-start_time}s")
        print(f"Detection results length: {len(res)}")
        
        # --- Assign Detection Results with Alignment ---
        print(f"Detection results length: {len(res)}, DataFrame index length: {len(df_eval)}")

        # Check if sequence_length is valid and > 0 before proceeding
        if sequence_length is None or sequence_length <= 0:
            raise RuntimeError("Cannot assign detection results: sequence_length is unknown or invalid.")

        if len(res) == len(df_eval):
            # Lengths already match (e.g., sequence_length was 1 or model handled padding)
            print("Assigning detection results directly (lengths match).")
            df_eval["is_anomaly"] = res.values if isinstance(res, pd.Series) else res
        elif len(res) < len(df_eval):
            # Length of results is shorter - likely due to sequence window. Pad the beginning.
            padding_len = len(df_eval) - len(res)
            expected_padding = sequence_length - 1
            if padding_len != expected_padding:
                warnings.warn(
                    f"Length difference ({padding_len}) between df_eval and detection results "
                    f"does not match expected padding ({expected_padding}) based on sequence_length={sequence_length}. "
                    "Alignment might be incorrect.", RuntimeWarning
                )

            print(f"Padding detection results with {padding_len} 'False' values at the beginning.")
            # Default value for padding (usually False for anomaly detection)
            padding = [False] * padding_len

            # Ensure 'res' is in a list format for concatenation
            if isinstance(res, (pd.Series, np.ndarray)):
                res_list = res.tolist()
            elif isinstance(res, list):
                res_list = res
            else:
                # Attempt conversion for other types, but issue a warning
                warnings.warn(f"Unexpected type for detection result: {type(res)}. Attempting conversion to list.", RuntimeWarning)
                try:
                    res_list = list(res)
                except TypeError:
                    raise TypeError(f"Cannot convert detection result of type {type(res)} to list for padding.")

            # Combine padding and results
            combined_values = padding + res_list

            # Final check - should always match now if padding was correct
            if len(combined_values) != len(df_eval):
                raise ValueError(f"Internal Error: Padded values length ({len(combined_values)}) still mismatch index length ({len(df_eval)}).")

            df_eval["is_anomaly"] = combined_values
        else: # len(res) > len(df_eval)
            # This is unexpected - the result should not be longer than the input frame index
            raise ValueError(f"Detection result length ({len(res)}) is greater than DataFrame index length ({len(df_eval)}). Cannot assign.")
        # --- End Assign Detection Results ---
        # --- End Anomaly Detection ---

        # --- Anomaly Update and Evaluation ---
        anomaly_df = df_eval[df_eval["is_anomaly"] == True]
        arr = [datetime.fromtimestamp(timestamp) for timestamp in anomaly_df["timestamp"]]
        arr = [f'\'{str(time)}+00\'' for time in arr]
        api.update_anomalies(name, arr)
        evaluation_results = evaluate_classification(df_eval) # Use df_eval with 'is_anomaly'
        print("Evaluation Results:")
        print(evaluation_results)
        # --- End Anomaly Update ---
            
        # ============================================
        # --- MODULAR XAI INTEGRATION ---
        # ============================================
        if xai_params and isinstance(xai_params, list) and model_wrapper is not None:
            print(f"Processing {len(xai_params)} XAI method(s)...") # Log how many methods

            max_bg_samples = 25000
            
            # Loop through each configuration dictionary in the list
            for xai_config in xai_params:
                method_name = xai_config.get("method")
                settings = xai_config.get("settings", {})

                shap_method = None
                if method_name == 'ShapExplainer':
                    try:
                        shap_method = settings.get('shap_method', 'kernel')
                    except:
                        shap_method = 'kernel'
                
                output_dir = "/data" # Ensure this path is accessible/writable in Docker

                if not method_name or method_name == "none":
                    print("XAI method not specified in parameters. Skipping XAI.")
                elif TimeSeriesExplainer is None or ut.dataframe_to_sequences is None:
                    print("XAI components not available (import failed). Skipping XAI.")
                else:
                    xai_methods_to_run = [method_name] # Explain only the selected method
                    output_dir = "/data"
                    os.makedirs(output_dir, exist_ok=True)

                    # Check prerequisites
                    if (TimeSeriesExplainer is not None and
                            ut.dataframe_to_sequences is not None and
                            sequence_length is not None and
                            model_wrapper is not None and
                            # Make sure plot handlers are imported
                            x.process_and_plot_shap is not None):

                        print("\n--- Starting XAI Initialization & Execution ---")
                        try:
                            # 1. Prepare Common Background Data
                            print(f"Preparing background data...")
                            background_data_np = ut.dataframe_to_sequences(
                                df=training_features_df, sequence_length=sequence_length,
                                feature_cols=feature_columns
                            )

                            if shap_method == 'tree':
                                max_bg_samples = len(background_data_np)

                            if len(background_data_np) > max_bg_samples:
                                print(f"Sampling background data down to {max_bg_samples} instances.")
                                indices = np.random.choice(len(background_data_np), max_bg_samples, replace=False)
                                background_data_np = background_data_np[indices]

                            if background_data_np.size == 0:
                                print("Warning: Background data generation resulted in empty array. Skipping XAI.")
                                raise ValueError("Empty background data") # Stop XAI if background fails
                            
                            # --- Extract Corresponding 1D Background Labels ---
                            # The label for sequence `i` corresponds to the label at the end
                            # of that sequence in the original training DataFrame.
                            num_sequences = background_data_np.shape[0]
                            # Calculate the indices in the original training DataFrame corresponding to the end of each sequence
                            end_indices = [i + sequence_length - 1 for i in range(num_sequences)]
                            # Ensure indices are within bounds of the training labels DataFrame
                            valid_end_indices = [idx for idx in end_indices if idx < len(training_df_with_labels)]

                            if len(valid_end_indices) != num_sequences:
                                warnings.warn(f"Mismatch between number of generated sequences ({num_sequences}) and valid end indices ({len(valid_end_indices)}). Check sequence generation logic.", RuntimeWarning)
                                # Handle this case - perhaps trim background_data_np? For now, trim labels.
                                background_data_np = background_data_np[:len(valid_end_indices)] # Trim sequences if indices are out of bounds

                            # Extract labels using the valid indices from the DataFrame containing labels
                            try:
                                background_outcomes_np = training_df_with_labels[actual_label_col].iloc[valid_end_indices].values
                                print(f"Extracted {len(background_outcomes_np)} background labels corresponding to sequences.")
                            except KeyError:
                                raise KeyError(f"Label column '{actual_label_col}' not found in training_df_with_labels.")
                            except IndexError:
                                raise IndexError("Error accessing labels using calculated end indices. Check sequence alignment.")
                            # --- END NEW ---

                            shap_method = None
                            if method_name == 'ShapExplainer':
                                try:
                                    shap_method = settings.get('shap_method', 'kernel')
                                except:
                                    shap_method = 'kernel'

                            # 2. Initialize TimeSeriesExplainer
                            try:
                                ts_explainer = TimeSeriesExplainer(
                                    model=model_wrapper, # The wrapper (e.g., ModelWrapperForXAI)
                                    background_data=background_data_np, # 3D Features for background
                                    background_outcomes=background_outcomes_np,
                                    feature_names=feature_columns,      # Base feature names
                                    mode='classification',
                                    # --- Pass DiCE specific context as kwargs ---
                                    training_df_for_dice=training_df_with_labels,
                                    outcome_name_for_dice=actual_label_col,
                                    continuous_features_for_dice=continuous_features_list,
                                    # --- Shap Explainer Method ---
                                    shap_method=shap_method
                                )
                            except Exception as e:
                                print(f"Failed to initialize TimeSeriesExplainer: {e}")
                                # Handle error appropriately, maybe skip XAI
                                ts_explainer = None

                            # 3. Prepare Instances for Explanation
                            # Choose data: anomalies if available, otherwise fallback
                            data_source_for_explanation = all_features_df_with_labels if not all_features_df_with_labels.empty else anomaly_feature_df
                            print(f"Running explanation on columns: {data_source_for_explanation.columns.values} with a length of {len(data_source_for_explanation)}")

                            # Create 3D feature array for explanation
                            instances_to_explain_np = ut.dataframe_to_sequences(
                                df=data_source_for_explanation, sequence_length=sequence_length,
                                feature_cols=feature_columns
                            )

                            # Limit number of explanations AFTER generating sequences
                            n_explain_max = settings.get("n_explain_max", 10) # Example: get max explain from settings
                            num_instances_available = instances_to_explain_np.shape[0]

                            if num_instances_available == 0:
                                print("Warning: No sequences generated for explanation. Skipping XAI method.")
                                continue # Skip to the next method if no instances

                            num_instances_to_explain = min(num_instances_available, n_explain_max)
                            if num_instances_available > n_explain_max:
                                print(f"Limiting explanation instances from {num_instances_available} to {num_instances_to_explain}.")
                                instances_to_explain_np = instances_to_explain_np[:num_instances_to_explain]

                            print(f"Prepared {instances_to_explain_np.shape[0]} instances (features) for explanation.")

                            # --- Extract Corresponding True Labels ---
                            original_labels_for_handler = None # Initialize
                            try:
                                # Calculate end indices in the source DataFrame for the sequences being explained
                                # Assumes sequential generation: 0th sequence ends at index (seq_len - 1),
                                # 1st sequence ends at index (seq_len), etc.
                                start_index_in_df = sequence_length - 1
                                end_indices_for_explanation = list(range(start_index_in_df, start_index_in_df + num_instances_to_explain))

                                # Ensure indices are valid for the source DataFrame length
                                max_source_index = len(data_source_for_explanation) - 1
                                if not end_indices_for_explanation or end_indices_for_explanation[-1] > max_source_index:
                                    raise IndexError(f"Calculated end index ({end_indices_for_explanation[-1]}) exceeds source data length ({max_source_index}). Check sequence generation.")

                                # Extract labels using calculated indices
                                if actual_label_col not in data_source_for_explanation.columns:
                                    raise KeyError(f"Label column '{actual_label_col}' not found in data_source_for_explanation DataFrame.")

                                original_labels_for_handler = data_source_for_explanation[actual_label_col].iloc[end_indices_for_explanation].values
                                print(f"Successfully extracted {len(original_labels_for_handler)} true labels for explained instances.")

                                # Sanity check lengths
                                if len(original_labels_for_handler) != num_instances_to_explain:
                                    warnings.warn("Length mismatch between extracted labels and instances to explain after indexing. Check logic.", RuntimeWarning)
                                    # Attempt to recover if possible, otherwise might need to skip
                                    min_len = min(len(original_labels_for_handler), num_instances_to_explain)
                                    original_labels_for_handler = original_labels_for_handler[:min_len]
                                    instances_to_explain_np = instances_to_explain_np[:min_len]
                                    print(f"Adjusted instances/labels to length {min_len} due to mismatch.")

                            except (KeyError, IndexError, Exception) as label_err:
                                print(f"ERROR extracting original labels for handler: {label_err}")
                                print("Proceeding without original labels for the handler.")
                                # original_labels_for_handler remains None

                            # 4. Define Plot Handlers Dictionary (mapping names to functions)
                            plot_handlers = { "ShapExplainer": x.process_and_plot_shap, 
                                            "LimeExplainer": x.process_and_plot_lime,
                                            "DiceExplainer": x.process_and_plot_dice }

                            # 5. Loop Through XAI Methods defined in xai_methods_to_run list
                            for method_name in xai_methods_to_run: # Iterate over the LIST of method names
                                print(f"\n===== Running Method: {method_name.upper()} =====")
                                try:
                                    explainer_object = ts_explainer._get_or_initialize_explainer(method_name)

                                    # Get base config for this method, provide empty dict if not found
                                    method_config = settings
                                    print(f"Using configuration for {method_name}: {method_config}")

                                    features_to_vary = method_config.get('features_to_vary', [])
                                    if len(features_to_vary) == 0:
                                        features_to_vary = feature_columns

                                    # --- Method Specific Logic ---
                                    if method_name == "DiceExplainer":
                                        # Prepare DiCE runtime args
                                        dice_runtime_kwargs = {
                                            'total_CFs': method_config.get('total_CFs', 4),
                                            'desired_class': method_config.get('desired_class', 'opposite'),
                                            'features_to_vary': features_to_vary,
                                            # Add other things if specified in frontend
                                        }
                                        print(f"DiCE Runtime Params: {dice_runtime_kwargs}")
                                        current_instances_np = instances_to_explain_np # DiCE explain handles batch

                                        start_explain_time = time.perf_counter()
                                        xai_results = ts_explainer.explain(
                                            instances_to_explain=current_instances_np,
                                            method_name=method_name, # "DiceExplainer"
                                            **dice_runtime_kwargs
                                        )
                                        end_explain_time = time.perf_counter()
                                        print(f"DICE explanation took {end_explain_time - start_explain_time:.2f}s")

                                        # Call DiCE Handler
                                        handler_func = plot_handlers.get(method_name)
                                        if handler_func:
                                            handler_func(results=xai_results, explainer_object=explainer_object, instances_explained=instances_to_explain_np, original_labels=original_labels_for_handler, feature_names=feature_columns, sequence_length=sequence_length, output_dir=output_dir, mode='classification', job_name=name)
                                        else: print("No plot handler for DiCE")

                                    elif method_name == "ShapExplainer":
                                        # Prepare SHAP runtime kwargs
                                        n_flat_features = sequence_length * len(feature_columns)
                                        n_samples_shap = method_config.get('nsamples', 50) # Use config or default
                                        k_features = method_config.get('l1_reg_k_features', 20)
                                        l1_reg_shap = f'num_features({k_features})' if n_samples_shap < n_flat_features else 'auto'
                                        shap_runtime_kwargs = {
                                            'nsamples': n_samples_shap,
                                            'l1_reg': l1_reg_shap,
                                            # Add other SHAP explain kwargs if needed from method_config
                                        }
                                        print(f"SHAP Runtime Params: {shap_runtime_kwargs}")

                                        # Explain ALL selected instances in one batch for SHAP
                                        start_explain_time = time.perf_counter()
                                        xai_results = ts_explainer.explain(
                                            instances_to_explain=instances_to_explain_np,
                                            method_name=method_name, # Should be "shap"
                                            **shap_runtime_kwargs # Pass SHAP specific args
                                        )
                                        end_explain_time = time.perf_counter()
                                        print(f"SHAP explanation took {end_explain_time - start_explain_time:.2f}s")

                                        handler_func = plot_handlers.get(method_name)
                                        if handler_func:
                                            print(f"Calling plot handler for SHAP...")
                                            handler_args = { # Prepare args dict
                                                "results": xai_results, "explainer_object": explainer_object,
                                                "instances_explained": instances_to_explain_np,
                                                "feature_names": feature_columns, "sequence_length": sequence_length,
                                                "output_dir": output_dir, "mode": ts_explainer.mode, "job_name": name
                                            }
                                            handler_func(**handler_args) # Unpack dict
                                        else: print(f"No plot handler defined for SHAP.")

                                    elif method_name == "LimeExplainer":
                                        print(f"Configuring LIME for {len(instances_to_explain_np)} instance(s)...")
                                        # Prepare LIME runtime kwargs from config
                                        num_features = method_config.get('num_features', 15)
                                        num_samples = method_config.get('num_samples', 1000)
                                        lime_runtime_kwargs = {
                                            'num_features': num_features,
                                            'num_samples': num_samples
                                            # Add other LIME explain_instance kwargs (like top_labels) if configured
                                        }
                                        print(f"LIME Runtime Params: {lime_runtime_kwargs}")

                                        # INNER LOOP FOR LIME INSTANCES
                                        for instance_idx in range(len(instances_to_explain_np)):
                                            print(f"--- Explaining Instance {instance_idx} with LIME ---")
                                            current_instance_np = instances_to_explain_np[instance_idx : instance_idx + 1]
                                            if current_instance_np.size == 0: continue

                                            try:
                                                start_lime_time = time.perf_counter()
                                                # Explain this single instance slice
                                                lime_explanation_object = ts_explainer.explain(
                                                    instances_to_explain=current_instance_np,
                                                    method_name='LimeExplainer', # Method name is "lime"
                                                    **lime_runtime_kwargs # Pass LIME specific args
                                                )
                                                end_lime_time = time.perf_counter()
                                                print(f"LIME explanation for instance {instance_idx} took {end_lime_time - start_lime_time:.2f}s")

                                                # Call the LIME handler
                                                handler_func = plot_handlers.get(method_name)
                                                if handler_func:
                                                    print(f"Calling plot handler for LIME instance {instance_idx}...")
                                                    handler_args = { # Prepare args dict
                                                        "results": lime_explanation_object, "explainer_object": explainer_object,
                                                        "instances_explained": current_instance_np,
                                                        "feature_names": feature_columns, "sequence_length": sequence_length,
                                                        "output_dir": output_dir, "mode": ts_explainer.mode,
                                                        "instance_index": instance_idx, "job_name": name
                                                    }
                                                    handler_func(**handler_args) # Unpack dict
                                                else: print(f"No plot handler defined for LIME.")

                                            except Exception as lime_instance_err:
                                                print(f"ERROR during LIME explanation/plotting for instance {instance_idx}: {lime_instance_err}")

                                        print(f"--- Finished LIME Explanations ---")
                                        # END INNER LOOP FOR LIME

                                    else:
                                        print(f"Skipping unknown or unhandled XAI method: {method_name}")

                                except Exception as explain_err:
                                    print(f"ERROR during explanation/plotting setup for method '{method_name}': {explain_err}")

                            # End main XAI method loop

                        except Exception as xai_init_err:
                            print(f"ERROR during XAI initialization or data preparation: {xai_init_err}")
                            import traceback
                            traceback.print_exc()
                    else:
                        skip_reasons = []
                        # ... (collect skip reasons) ...
                        print(f"Skipping XAI step. Prerequisites not met. Reasons: {', '.join(skip_reasons)}")
                    # ============================================
                    # --- END MODULAR XAI INTEGRATION ---
                    # ============================================
        else:
            print("No xai method is chosen. Skipping explanations")

        sys.stdout.flush()
    else:
        return 0

# Starts processing of dataset as a stream
def run_stream(db_conn_params, 
               model: str, 
               path: str, 
               name: str, 
               speedup: int, 
               inj_params: dict=None, 
               debug=False) -> None:
    print("Starting Stream-job!")
    sys.stdout.flush()

    if inj_params is not None:
        anomaly_settings = []  # Create a list to hold AnomalySetting objects
        for params in inj_params:  # Iterate over the list of anomaly dictionaries
            anomaly = AnomalySetting(
                params.get("anomaly_type", None),
                int(params.get("timestamp", None)),
                int(params.get("magnitude", None)),
                int(params.get("percentage", None)),
                params.get("columns", None),
                params.get("duration", None)
            )
            anomaly_settings.append(anomaly)  # Add the AnomalySetting object to the list
        stream_job = Job(filepath=path, anomaly_settings=anomaly_settings, simulation_type="stream", speedup=speedup, table_name=name, debug=debug)
    else:
        print("Should not inject anomaly.")
        stream_job = Job(filepath=path, simulation_type="stream", speedup=speedup, table_name=name, debug=debug)

    sim_engine = se()
    sim_engine.main(db_conn_params, stream_job)

def single_point_detection(api, simulation_thread, model, name, path):
    
    model_instance = get_model(model)
    df = pd.read_csv(path)
    model_instance.run(df)

    while not api.table_exists(name):
        time.sleep(1)
    
    
    timestamp = datetime.fromtimestamp(0)
    
    while simulation_thread.is_alive():
        df = api.read_data(datetime.fromtimestamp(0), name)
        timestamp = df["timestamp"].iloc[-1].to_pydatetime()
        print(df["timestamp"].iloc[-1])

        df["timestamp"] = df["timestamp"].apply(map_to_timestamp)
        df["timestamp"] = df["timestamp"].astype(float)

        res = model_instance.detect(df.iloc[:, :-2])
        df["is_anomaly"] = res
        
        anomaly_df = df[df["is_anomaly"] == True]
        arr = [datetime.fromtimestamp(timestamp) for timestamp in anomaly_df["timestamp"]]
        arr = [f'\'{str(time)}+00\'' for time in arr]
        
        api.update_anomalies(name, arr)
    
        time.sleep(1)


# Returns a list of models implemented in MODEL_DIRECTORY
def get_models() -> list:
    models = []
    for path in os.listdir(MODEL_DIRECTORY):
        file_path = MODEL_DIRECTORY + "/" + path
        if os.path.isfile(file_path):
            model_name = path.split(".")[0]
            models.append(model_name)

    # Removing the __init__, setup files and the .env file
    models.remove("")
    models.remove("model_interface")
    models.remove("__init__")
    models.remove("setup")
    models.remove("get_model")
    models.remove("model_wrapper")
    
    return models

# Returns a list of XAI mthods implemented in XAI_METHOD_DIRECTORY
def get_xai_methods() -> list:
    methods = []
    for path in os.listdir(XAI_METHOD_DIRECTORY):
        if os.path.isfile(os.path.join(XAI_METHOD_DIRECTORY, path)):
            method_name = path.split(".")[0]
            methods.append(method_name)

    # Removing the __init__, setup files and the .env file
    methods.remove("__init__")
    methods.remove("dice_builder")

    return methods

# Returns a list of injection methods implemented in INJECTION_METHOD_DIRECTORY
def get_injection_methods() -> list:
    injection_methods = []

    for path in os.listdir(INJECTION_METHOD_DIRECTORY):
        if os.path.isfile(os.path.join(INJECTION_METHOD_DIRECTORY, path)):
            method_name = path.split(".")[0]
            injection_methods.append(method_name)

    injection_methods.remove("__init__")
    return injection_methods

# Fetching datasets from the dataset directory
def get_datasets() -> list:
    datasets = []
    for path in os.listdir(DATASET_DIRECTORY):
        file_path = DATASET_DIRECTORY + "/" + path
        if os.path.isfile(file_path):
            dataset = path
            datasets.append(dataset)

    return datasets

# Gets content of complete file to the backend
def import_dataset(conn: socket, path: str, timestamp_column: str) -> None:
    file = open(path, "w")
    data = conn.recv(1024).decode("utf-8")
    while data:
        file.write(data)
        data = conn.recv(1024).decode("utf-8")
    file.close()
    
    # Change the timestamp column name to timestamp and move it to the first column
    df = pd.read_csv(path)
    df.rename(columns={timestamp_column: "timestamp"}, inplace=True)
    cols = df.columns.tolist()
    cols.remove("timestamp")
    cols = ["timestamp"] + cols
    df = df[cols]
    df.to_csv(path, index=False)