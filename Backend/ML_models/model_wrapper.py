import traceback
import pandas as pd
import numpy as np
from typing import Any, List, Union, Literal # Import Literal
import warnings

class ModelWrapperForXAI:
    """
    Wraps a model with 'detect' and 'predict_proba' methods to provide
    '.predict()' and '.predict_proba()' interfaces for XAI tools (SHAP, LIME, DiCE).

    Handles different input requirements (DataFrame vs NumPy) of internal model methods
    and assumes the internal 'predict_proba' method returns a probability
    of anomaly P(anomaly) between 0 and 1.
    """
    def __init__(self,
                 actual_model_instance: Any,
                 feature_names: List[str],
                 # score_interpretation is kept for potential future use or info,
                 # but predict_proba now relies on predict_proba returning P(anomaly) directly.
                 score_interpretation: Literal['lower_is_anomaly', 'higher_is_anomaly'] = 'higher_is_anomaly'
                 ):
        """
        Args:
            actual_model_instance: Trained model instance with callable 'detect' & 'predict_proba'.
                                   'predict_proba' MUST return P(anomaly) in [0, 1].
            feature_names (List[str]): Feature names list corresponding to the last dim of input data.
            score_interpretation (str): Informational about how the original score was interpreted
                                        BEFORE being converted to P(anomaly) by predict_proba.
                                        Set to 'lower_is_anomaly' (e.g., OCSVM) or
                                        'higher_is_anomaly' (e.g., AE reconstruction error).
        """
        self._model = actual_model_instance
        self._feature_names = feature_names
        self._num_classes = 2 # Assuming binary classification (normal vs anomaly)
        self._score_interpretation = score_interpretation # Store for information

        # --- Checks ---
        if not all(hasattr(self._model, meth) and callable(getattr(self._model, meth))
                   for meth in ['detect', 'predict_proba']):
            raise AttributeError("Provided model must have callable 'detect' and 'predict_proba' methods.")
        # Ensure predict_proba exists and is callable
        if not hasattr(self._model, 'predict_proba') or not callable(getattr(self._model, 'predict_proba')):
             raise AttributeError("The wrapped model MUST have a callable 'predict_proba' method that returns P(anomaly).")

        if len(self._feature_names) == 0: raise ValueError("Feature names list cannot be empty.")
        if self._score_interpretation not in ['lower_is_anomaly', 'higher_is_anomaly']:
            raise ValueError("score_interpretation must be 'lower_is_anomaly' or 'higher_is_anomaly'")
        if not hasattr(self._model, 'sequence_length'):
             warnings.warn("Wrapped model lacks 'sequence_length' attribute. Assuming sequence length of 1 if input is 2D.", RuntimeWarning)
        # --- End Checks ---

        print(f"ModelWrapperForXAI initialized for model type: {type(self._model).__name__}")
        print(f"Underlying score interpretation (used by predict_proba): '{self._score_interpretation}'")

    @property
    def sequence_length(self):
        """Returns the sequence length from the underlying model, or None if not defined."""
        return getattr(self._model, 'sequence_length', None)

    @property
    def model(self) -> Any:
        """Provides access to the underlying model instance."""
        return self._model

    # --- Private helper to call internal model method ---
    def _call_internal_method(self, X_input_data: np.ndarray, internal_method_name: str) -> np.ndarray:
        """
        Calls the specified method on the wrapped model, ensuring input is NumPy
        and returning NumPy array. Handles potential errors.

        Args:
            X_input_data (np.ndarray): The input data (should be 3D for sequence models).
            internal_method_name (str): The name of the method to call ('detect' or 'predict_proba').

        Returns:
            np.ndarray: The results from the internal method as a NumPy array.
        """
        internal_method = getattr(self._model, internal_method_name)
        # print(f"Wrapper: Calling internal '{internal_method_name}' with input shape {X_input_data.shape}...") # Debug print
        try:
            # Assuming internal methods now correctly handle the 3D NumPy input
            results = internal_method(X_input_data)

            # Ensure output is a NumPy array
            if isinstance(results, (list, pd.Series)):
                results_np = np.array(results)
            elif isinstance(results, np.ndarray):
                results_np = results
            elif isinstance(results, (int, float, bool)): # Handle single value output
                 results_np = np.array([results])
            else:
                 warnings.warn(f"Internal method '{internal_method_name}' returned unexpected type {type(results)}. Attempting conversion.", RuntimeWarning)
                 try:
                     results_np = np.array(results)
                 except Exception as conv_err:
                     print(f"ERROR: Could not convert result of internal method '{internal_method_name}' to NumPy array: {conv_err}")
                     raise TypeError(f"Internal method '{internal_method_name}' failed to return convertible type.") from conv_err

            # print(f"Wrapper: Internal '{internal_method_name}' returned array shape {results_np.shape}") # Debug print
            return results_np

        except Exception as e:
            print(f"ERROR in ModelWrapper calling internal model.{internal_method_name} with input shape {X_input_data.shape}: {e}")
            traceback.print_exc() # Print full traceback
            # Return array of NaNs matching expected output length (number of samples)
            n_out = X_input_data.shape[0] if X_input_data.ndim >= 1 else 1
            return np.full(n_out, np.nan)


    def predict(self, X_np_3d: np.ndarray) -> np.ndarray:
        """
        Provides the .predict() interface for XAI tools.
        Calls the internal model's 'detect' method.

        Args:
            X_np_3d (np.ndarray): Input data, expected to be 3D (samples, seq_len, features).

        Returns:
            np.ndarray: Binary predictions (0 or 1), reshaped to (samples, 1).
        """
        if X_np_3d.ndim != 3:
             warnings.warn(f"Wrapper 'predict' received input with {X_np_3d.ndim} dimensions. Expected 3D. Proceeding, but internal model might fail.", RuntimeWarning)

        # Call internal 'detect'
        raw_predictions = self._call_internal_method(X_np_3d, 'detect')

        # --- Process raw predictions ---
        # Ensure float type for potential NaN checks, default fill is int 0
        results_np = np.array(raw_predictions, dtype=float)
        nan_mask = np.isnan(results_np)
        if np.any(nan_mask):
             warnings.warn(f"'detect' method returned NaNs. Filling with 0.", RuntimeWarning)
             results_np[nan_mask] = 0 # Default non-anomaly prediction for NaNs

        # Convert boolean results to int (True->1, False->0) if necessary
        # Check the type *after* handling NaNs
        if pd.api.types.is_bool_dtype(results_np):
            results_np = results_np.astype(int)
        elif not pd.api.types.is_numeric_dtype(results_np):
             warnings.warn(f"'detect' method returned non-numeric, non-boolean type {results_np.dtype} after NaN handling. Attempting conversion to int, might fail.", RuntimeWarning)
             try:
                 results_np = results_np.astype(int)
             except ValueError:
                 warnings.warn(f"Could not convert 'detect' results to int. Filling with 0.", RuntimeWarning)
                 results_np.fill(0) # Fallback if conversion fails

        # Ensure output is 2D (samples, 1)
        if results_np.ndim == 1:
            results_np = results_np[:, np.newaxis]
        elif results_np.ndim == 0: # Handle scalar output
            results_np = np.array([[results_np.item()]])
        elif results_np.ndim > 2 or (results_np.ndim == 2 and results_np.shape[1] != 1):
             warnings.warn(f"'detect' method returned unexpected shape {results_np.shape} after processing. Attempting to reshape/select first column.", RuntimeWarning)
             # Attempt to recover if possible, e.g., take first column if multiple outputs
             if results_np.ndim >= 2 and results_np.shape[1] > 1:
                 results_np = results_np[:, 0:1] # Take first column, keep 2D
             else: # Cannot easily recover, return default shape
                 results_np = np.zeros((X_np_3d.shape[0], 1), dtype=int)


        # print(f"ModelWrapper.predict: Returning shape {results_np.shape}") # Debug print
        return results_np.astype(int) # Ensure integer output

    def predict_proba(self, X_input: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Provides the .predict_proba() interface for XAI tools (SHAP, LIME, DiCE).
        Handles input type (DataFrame/NumPy) and shape (2D/3D), calls the internal
        model's 'predict_proba' method, and returns the resulting probabilities.

        Args:
            X_input (Union[np.ndarray, pd.DataFrame]): Input data. Can be:
                - 3D NumPy array (samples, seq_len, features)
                - 2D NumPy array (samples, features) -> reshaped to (samples, 1, features)
                - 2D NumPy array (samples, seq_len * features) -> reshaped to 3D
                - pandas DataFrame (samples, features or seq_len * features) -> converted to NumPy and reshaped

        Returns:
            np.ndarray: Probabilities array of shape (samples, 2), where column 0 is
                        P(normal) and column 1 is P(anomaly). Returns neutral
                        probabilities [0.5, 0.5] for invalid inputs or internal errors.
        """
        print(f"\n--- Wrapper predict_proba called with input type {type(X_input)} ---")

        # -----------------------------------
        # Input Conversion and Reshaping 
        # -----------------------------------
        if isinstance(X_input, pd.DataFrame):
            print(f"DEBUG Wrapper predict_proba: Received DataFrame shape {X_input.shape}. Converting to NumPy.")
            X_np_input = X_input.to_numpy()
            print(f"DEBUG Wrapper predict_proba: Converted DataFrame to NumPy shape {X_np_input.shape}.")
        elif isinstance(X_input, np.ndarray):
            X_np_input = X_input  # Already a NumPy array
            print(f"DEBUG Wrapper predict_proba: Received NumPy input shape {X_np_input.shape}.")
        else:
            raise TypeError(f"ModelWrapperForXAI.predict_proba expects NumPy array or pandas DataFrame, got {type(X_input)}")

        expected_features = len(self._feature_names)
        X_to_pass_internally = None  # This will hold the 3D array for the internal model call
        n_samples_in = 0 # Track original number of samples before processing

        # --- Input Shape Handling ---
        if X_np_input.ndim >= 1:
            n_samples_in = X_np_input.shape[0]
        else:
            n_samples_in = 1 # Handle 0-dim edge case

        if X_np_input.ndim == 3:
            # Input is already 3D. Assume it's correct for models trained on 3D numpy.
            # The internal model's XAI predict_proba should validate seq_len/features.
            if X_np_input.shape[-1] == expected_features:
                X_to_pass_internally = X_np_input
                print(f"DEBUG Wrapper predict_proba: Using 3D input shape {X_np_input.shape}.")
            else:
                warnings.warn(f"Wrapper predict_proba received 3D input with unexpected feature count {X_np_input.shape[-1]}. Expected {expected_features}. Returning neutral probabilities.", RuntimeWarning)
                return np.full((n_samples_in, 2), 0.5)

        elif X_np_input.ndim == 2:
            # Input is 2D.
            # If the model was trained on DataFrame, it expects 2D.
            # If the model was trained on NumPy, its XAI predict_proba might expect 2D (flattened) or need reshaping - let internal handle it.
            # ---> Pass the 2D array directly. The internal XAI predict_proba should know what to do based on self.input_type.
            X_to_pass_internally = X_np_input
            print(f"DEBUG Wrapper predict_proba: Reshaped 2D input ({X_np_input.shape}) to 3D ({X_to_pass_internally.shape}) assuming seq_len=1.")
        elif X_np_input.ndim == 1:
            # Handle single sample 1D input. Reshape to 2D (1, n_features) for consistency.
            print(f"DEBUG Wrapper predict_proba: Reshaping 1D input ({X_np_input.shape}) to 2D (1, {X_np_input.shape[0]})")
            # Validate feature count for 1D case
            if len(X_np_input) == expected_features:
                X_to_pass_internally = X_np_input.reshape(1, -1)
            else:
                warnings.warn(f"Wrapper predict_proba received 1D input with unexpected feature count {len(X_np_input)}. Expected {expected_features}. Returning neutral probabilities.", RuntimeWarning)
                return np.full((n_samples_in, 2), 0.5)
        else:
            warnings.warn(f"Wrapper predict_proba received unexpected input shape {X_np_input.shape}. Expected 1D, 2D or 3D. Returning neutral probabilities.", RuntimeWarning)
            return np.full((n_samples_in, 2), 0.5)

        # Ensure X_to_pass_internally is assigned
        if X_to_pass_internally is None:
            # This case might be redundant now but kept for safety
            warnings.warn("Wrapper predict_proba: Failed to process input shape. Returning neutral probabilities.", RuntimeWarning)
            return np.full((n_samples_in, 2), 0.5)

        # ----------------------------------------------------------------------
        # Call Internal Model's predict_proba
        # ----------------------------------------------------------------------
        print(f"DEBUG Wrapper predict_proba: Calling internal 'predict_proba' with processed shape {X_to_pass_internally.shape}")
        try:
            # Assuming the internal method now returns shape (n_samples, 2) -> [P(normal), P(anomaly)]
            internal_probabilities = self._call_internal_method(X_to_pass_internally, 'predict_proba')
            print(f"DEBUG Wrapper predict_proba: Internal 'predict_proba' returned shape {internal_probabilities.shape}")
        except Exception as e:
            warnings.warn(f"Wrapper predict_proba: Error calling internal 'predict_proba': {e}. Returning neutral probabilities.", RuntimeWarning)
            # Use n_samples_in here as internal call failed before producing output
            return np.full((n_samples_in, 2), 0.5)

        # ----------------------------------------------------------------------
        # Process the Result from Internal predict_proba
        # ----------------------------------------------------------------------
        n_samples_out = internal_probabilities.shape[0] if internal_probabilities.ndim >= 1 else 0
        expected_cols = 2

        # Initialize final array, defaulting to neutral probabilities [0.5, 0.5]
        # Use n_samples_in for the initial shape to match input length expectation
        probabilities = np.full((n_samples_in, expected_cols), 0.5)

        # --- Basic validation of the internal output shape ---
        if internal_probabilities.ndim != 2 or internal_probabilities.shape[1] != expected_cols:
            warnings.warn(f"Wrapper predict_proba: Internal 'predict_proba' returned unexpected shape {internal_probabilities.shape}. Expected ({n_samples_out}, {expected_cols}). Returning neutral probabilities.", RuntimeWarning)
            # Keep the default neutral probabilities initialized above

        # --- Check for length mismatch ---
        elif n_samples_out != n_samples_in:
            warnings.warn(f"Wrapper predict_proba: Internal 'predict_proba' returned {n_samples_out} samples, but input had {n_samples_in}. Padding/truncating output.", RuntimeWarning)
            # Adjust internal_probabilities to match input length before further processing
            if n_samples_out < n_samples_in:
                padded_internal = np.full((n_samples_in, expected_cols), 0.5) # Pad with neutral
                padded_internal[:n_samples_out] = internal_probabilities
                internal_probabilities = padded_internal
            elif n_samples_out > n_samples_in:
                internal_probabilities = internal_probabilities[:n_samples_in]
            # Now internal_probabilities has shape (n_samples_in, 2)

        # --- Process Valid Probabilities ---
        # Handle potential NaNs or invalid values from the internal calculation
        # A row is invalid if *any* value in it is NaN or infinite
        invalid_row_mask = np.isnan(internal_probabilities).any(axis=1) | np.isinf(internal_probabilities).any(axis=1)

        if np.any(invalid_row_mask):
            warnings.warn("Wrapper predict_proba: NaNs/Infs detected in probabilities returned by internal 'predict_proba'. Corresponding rows set to [0.5, 0.5].", RuntimeWarning)
            # Rows with NaNs/Infs will keep the default [0.5, 0.5] initialized above

        # Process only valid (non-NaN/Inf) rows
        valid_row_mask = ~invalid_row_mask
        if np.any(valid_row_mask):
            print(f"DEBUG Wrapper predict_proba: Assigning {np.sum(valid_row_mask)} valid probability rows directly.")

            # --- THIS IS THE KEY CHANGE ---
            # Directly assign the valid probabilities from the internal result
            # No score_interpretation logic needed here anymore.
            probabilities[valid_row_mask] = internal_probabilities[valid_row_mask]
            # --- End Key Change ---

            # Optional: Clip probabilities just in case internal model returns values slightly outside [0,1]
            np.clip(probabilities[valid_row_mask], 0.0, 1.0, out=probabilities[valid_row_mask])

            # Optional: Re-normalize if sums are slightly off due to clipping/precision
            sums = np.sum(probabilities[valid_row_mask], axis=1, keepdims=True)
            # Avoid division by zero if sum is zero (shouldn't happen if clipped correctly)
            sums[sums < 1e-9] = 1.0 # Set sum to 1 if it's near zero
            probabilities[valid_row_mask] /= sums

        # --- Final Validation ---
        # Check if probabilities sum to 1 (within tolerance) for valid rows
        if np.any(valid_row_mask):
            final_sums = np.sum(probabilities[valid_row_mask], axis=1)
            if not np.allclose(final_sums, 1.0, atol=1e-6):
                warnings.warn("Wrapper predict_proba: Final probabilities do not sum close to 1 for some valid rows after processing. Check calculations.", RuntimeWarning)

        print(f"DEBUG Wrapper predict_proba: Returning final probabilities shape: {probabilities.shape}")
        # Print min/max/mean of final P(anomaly) column
        if probabilities.size > 0:
            print(f"DEBUG Wrapper predict_proba: Final P(anomaly) stats: min={np.min(probabilities[:, 1]):.4f}, max={np.max(probabilities[:, 1]):.4f}, mean={np.mean(probabilities[:, 1]):.4f}")

        # Ensure return shape matches n_samples_in
        if probabilities.shape[0] != n_samples_in:
            warnings.warn(f"Wrapper predict_proba: Final probability array shape {probabilities.shape} doesn't match input samples {n_samples_in}. Adjusting.", RuntimeWarning)
            # This should ideally not happen after the padding/truncation logic above
            # But as a final safeguard:
            final_probs_adjusted = np.full((n_samples_in, 2), 0.5)
            len_to_copy = min(probabilities.shape[0], n_samples_in)
            final_probs_adjusted[:len_to_copy] = probabilities[:len_to_copy]
            probabilities = final_probs_adjusted

        return probabilities