import traceback
import pandas as pd
import numpy as np
from typing import Any, List, Union, Literal # Import Literal
import warnings

class ModelWrapperForXAI:
    """
    Wraps a model with 'detect' and 'get_anomaly_score' methods to provide
    '.predict()' and '.predict_proba()' interfaces for XAI tools (SHAP, LIME, DiCE).

    Handles different input requirements (DataFrame vs NumPy) of internal model methods
    and assumes the internal 'get_anomaly_score' method returns a probability
    of anomaly P(anomaly) between 0 and 1.
    """
    def __init__(self,
                 actual_model_instance: Any,
                 feature_names: List[str],
                 # score_interpretation is kept for potential future use or info,
                 # but predict_proba now relies on get_anomaly_score returning P(anomaly) directly.
                 score_interpretation: Literal['lower_is_anomaly', 'higher_is_anomaly'] = 'higher_is_anomaly'
                 ):
        """
        Args:
            actual_model_instance: Trained model instance with callable 'detect' & 'get_anomaly_score'.
                                   'get_anomaly_score' MUST return P(anomaly) in [0, 1].
            feature_names (List[str]): Feature names list corresponding to the last dim of input data.
            score_interpretation (str): Informational about how the original score was interpreted
                                        BEFORE being converted to P(anomaly) by get_anomaly_score.
                                        Set to 'lower_is_anomaly' (e.g., OCSVM) or
                                        'higher_is_anomaly' (e.g., AE reconstruction error).
        """
        self._model = actual_model_instance
        self._feature_names = feature_names
        self._num_classes = 2 # Assuming binary classification (normal vs anomaly)
        self._score_interpretation = score_interpretation # Store for information

        # --- Checks ---
        if not all(hasattr(self._model, meth) and callable(getattr(self._model, meth))
                   for meth in ['detect', 'get_anomaly_score']):
            raise AttributeError("Provided model must have callable 'detect' and 'get_anomaly_score' methods.")
        # Ensure get_anomaly_score exists and is callable
        if not hasattr(self._model, 'get_anomaly_score') or not callable(getattr(self._model, 'get_anomaly_score')):
             raise AttributeError("The wrapped model MUST have a callable 'get_anomaly_score' method that returns P(anomaly).")

        if len(self._feature_names) == 0: raise ValueError("Feature names list cannot be empty.")
        if self._score_interpretation not in ['lower_is_anomaly', 'higher_is_anomaly']:
            raise ValueError("score_interpretation must be 'lower_is_anomaly' or 'higher_is_anomaly'")
        if not hasattr(self._model, 'sequence_length'):
             warnings.warn("Wrapped model lacks 'sequence_length' attribute. Assuming sequence length of 1 if input is 2D.", RuntimeWarning)
        # --- End Checks ---

        print(f"ModelWrapperForXAI initialized for model type: {type(self._model).__name__}")
        print(f"Underlying score interpretation (used by get_anomaly_score): '{self._score_interpretation}'")

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
            internal_method_name (str): The name of the method to call ('detect' or 'get_anomaly_score').

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
        'get_anomaly_score', and properly transforms the raw scores into probabilities
        based on the score_interpretation parameter.

        Args:
            X_input (Union[np.ndarray, pd.DataFrame]): Input data. Can be:
                - 3D NumPy array (samples, seq_len, features)
                - 2D NumPy array (samples, features) -> reshaped to (samples, 1, features)
                - 2D NumPy array (samples, seq_len * features) -> reshaped to 3D
                - pandas DataFrame (samples, features or seq_len * features) -> converted to NumPy and reshaped

        Returns:
            np.ndarray: Probabilities array of shape (samples, 2), where column 0 is
                        P(normal) and column 1 is P(anomaly).
        """
        print(f"\n--- predict_proba called with input type {type(X_input)} ---")

        # Convert DataFrame input to NumPy array
        if isinstance(X_input, pd.DataFrame):
            print(f"DEBUG predict_proba: Received DataFrame shape {X_input.shape}. Converting to NumPy.")
            X_np_input = X_input.to_numpy()
            print(f"DEBUG predict_proba: Converted DataFrame to NumPy shape {X_np_input.shape}.")
        elif isinstance(X_input, np.ndarray):
            X_np_input = X_input  # Already a NumPy array
            print(f"DEBUG predict_proba: Received NumPy input shape {X_np_input.shape}.")
        else:
            raise TypeError(f"ModelWrapperForXAI.predict_proba expects NumPy array or pandas DataFrame, got {type(X_input)}")

        expected_features = len(self._feature_names)
        X_processed_3d = None  # This will hold the 3D array for the internal model call

        # --- Input Shape Handling (Ensure X_processed_3d is 3D) ---
        if X_np_input.ndim == 3:
            # Input is already 3D (samples, seq_len, features)
            if X_np_input.shape[-1] == expected_features:
                X_processed_3d = X_np_input
                print(f"DEBUG predict_proba: Using 3D input shape {X_np_input.shape}.")
            else:
                # Feature mismatch in 3D input
                warnings.warn(f"Predict_proba received 3D input with unexpected feature count {X_np_input.shape[-1]}. Expected {expected_features}. Returning neutral probabilities.", RuntimeWarning)
                num_samples = X_np_input.shape[0]
                return np.full((num_samples, 2), 0.5)  # Return neutral probabilities

        elif X_np_input.ndim == 2:
            # Input is 2D (samples, features or samples, flat_features)
            if X_np_input.shape[-1] == expected_features:
                # Reshape 2D (samples, features) to 3D (samples, 1, features)
                X_processed_3d = X_np_input[:, np.newaxis, :]
                print(f"DEBUG predict_proba: Reshaped 2D input ({X_np_input.shape}) to 3D ({X_processed_3d.shape}) assuming seq_len=1.")
            else:
                # Check if it matches the *flattened* feature count
                current_sequence_length = self.sequence_length or 1  # Use 1 if not defined
                expected_flat_features = current_sequence_length * expected_features
                if X_np_input.shape[1] == expected_flat_features:
                    # Input looks like flattened sequence data (e.g., from LIME/DiCE)
                    # Reshape back to 3D: (samples, flat_features) -> (samples, seq_len, features)
                    num_samples = X_np_input.shape[0]
                    try:
                        X_processed_3d = X_np_input.reshape((num_samples, current_sequence_length, expected_features))
                        print(f"DEBUG predict_proba: Reshaped 2D flattened input ({X_np_input.shape}) to 3D ({X_processed_3d.shape}).")
                    except ValueError as reshape_err:
                        warnings.warn(f"Predict_proba failed to reshape 2D input ({X_np_input.shape}) to 3D ({num_samples}, {current_sequence_length}, {expected_features}). Error: {reshape_err}. Returning neutral probabilities.", RuntimeWarning)
                        return np.full((num_samples, 2), 0.5)
                else:
                    # Genuine feature mismatch in 2D input
                    warnings.warn(f"Predict_proba received 2D input with unexpected feature count {X_np_input.shape[1]}. Expected {expected_features} or {expected_flat_features} (flattened). Returning neutral probabilities.", RuntimeWarning)
                    num_samples = X_np_input.shape[0]
                    return np.full((num_samples, 2), 0.5)  # Return neutral probabilities

        else:
            # Unexpected input dimensions
            warnings.warn(f"Predict_proba received unexpected input shape {X_np_input.shape}. Expected 2D or 3D. Returning neutral probabilities.", RuntimeWarning)
            num_samples = X_np_input.shape[0] if X_np_input.ndim >= 1 else 1
            return np.full((num_samples, 2), 0.5)  # Return neutral probabilities

        # --- Call Internal Method to get Raw Anomaly Scores ---
        print(f"DEBUG predict_proba: Calling internal 'get_anomaly_score' with processed shape {X_processed_3d.shape}")
        raw_scores = self._call_internal_method(X_processed_3d, 'get_anomaly_score')
        print(f"DEBUG predict_proba: Internal 'get_anomaly_score' returned shape {raw_scores.shape}")
        
        # Optional: Print min/max/mean of raw scores
        if raw_scores.size > 0 and not np.all(np.isnan(raw_scores)):
            print(f"DEBUG predict_proba: Raw anomaly scores stats: min={np.nanmin(raw_scores):.4f}, max={np.nanmax(raw_scores):.4f}, mean={np.nanmean(raw_scores):.4f}")

        # --- Create Final 2D Probability Array [P(normal), P(anomaly)] ---
        n_samples_out = len(raw_scores)
        # Initialize final array, defaulting to neutral probabilities [0.5, 0.5]
        probabilities = np.full((n_samples_out, 2), 0.5)

        # Handle potential NaNs from the internal score calculation
        score_nan_mask = np.isnan(raw_scores)
        if np.any(score_nan_mask):
            warnings.warn("NaNs detected in raw anomaly scores returned by 'get_anomaly_score'. Corresponding rows set to [0.5, 0.5].", RuntimeWarning)

        # Process only valid (non-NaN) scores
        valid_scores_mask = ~score_nan_mask
        if np.any(valid_scores_mask):
            valid_raw_scores = raw_scores[valid_scores_mask]

            # --- IMPORTANT FIX: Transform raw scores to P(anomaly) based on score_interpretation ---
            # The previous implementation incorrectly assumed get_anomaly_score returned P(anomaly) directly
            
            # First, check if the scores might already be probabilities (in [0,1] range)
            scores_in_probability_range = np.all((valid_raw_scores >= 0) & (valid_raw_scores <= 1))
            
            # If all scores are in [0,1] range and the model's get_anomaly_score is documented to
            # return P(anomaly) directly, we can use them directly.
            if scores_in_probability_range:
                print("DEBUG predict_proba: Raw scores are in [0,1] range, might already be probabilities.")
                # But we still apply interpretation logic to be safe
            
            # Apply transformation based on score_interpretation
            if self._score_interpretation == 'lower_is_anomaly':
                # For models where lower scores mean more anomalous (e.g., OCSVM)
                # We need to transform to get P(anomaly)
                if scores_in_probability_range:
                    # If already in [0,1], might be 1-P(anomaly), so invert
                    prob_anomaly = 1.0 - valid_raw_scores
                else:
                    # Otherwise, we need to normalize and invert
                    # First find reasonable min/max values (avoiding outliers if possible)
                    if len(valid_raw_scores) > 10:
                        # With sufficient data, use percentiles to avoid extreme outliers
                        min_score = np.percentile(valid_raw_scores, 1)
                        max_score = np.percentile(valid_raw_scores, 99)
                    else:
                        # With limited data, use regular min/max
                        min_score = np.min(valid_raw_scores)
                        max_score = np.max(valid_raw_scores)
                    
                    # Normalize scores to [0,1] range, ensuring division by zero is avoided
                    score_range = max_score - min_score
                    if score_range > 1e-10:  # Non-zero range
                        normalized_scores = (valid_raw_scores - min_score) / score_range
                    else:  # All scores are approximately equal
                        normalized_scores = np.full_like(valid_raw_scores, 0.5)
                    
                    # Invert normalized scores: lower original score = higher P(anomaly)
                    prob_anomaly = 1.0 - normalized_scores
                    
            elif self._score_interpretation == 'higher_is_anomaly':
                # For models where higher scores mean more anomalous (e.g., AE reconstruction error)
                if scores_in_probability_range:
                    # If already in [0,1], might be P(anomaly) directly
                    prob_anomaly = valid_raw_scores
                else:
                    # Otherwise, normalize to get P(anomaly)
                    if len(valid_raw_scores) > 10:
                        # With sufficient data, use percentiles to avoid extreme outliers
                        min_score = np.percentile(valid_raw_scores, 1)
                        max_score = np.percentile(valid_raw_scores, 99)
                    else:
                        # With limited data, use regular min/max
                        min_score = np.min(valid_raw_scores)
                        max_score = np.max(valid_raw_scores)
                    
                    # Normalize scores to [0,1] range, ensuring division by zero is avoided
                    score_range = max_score - min_score
                    if score_range > 1e-10:  # Non-zero range
                        prob_anomaly = (valid_raw_scores - min_score) / score_range
                    else:  # All scores are approximately equal
                        prob_anomaly = np.full_like(valid_raw_scores, 0.5)
            else:
                # Should never happen due to initialization validation
                warnings.warn(f"Unknown score_interpretation '{self._score_interpretation}'. Using scores as-is.", RuntimeWarning)
                prob_anomaly = valid_raw_scores
                if not scores_in_probability_range:
                    # If not in [0,1], apply simple min-max normalization
                    min_score = np.min(valid_raw_scores)
                    max_score = np.max(valid_raw_scores)
                    score_range = max_score - min_score
                    if score_range > 1e-10:
                        prob_anomaly = (valid_raw_scores - min_score) / score_range
                    else:
                        prob_anomaly = np.full_like(valid_raw_scores, 0.5)

            # Ensure P(anomaly) is properly bounded in [0,1]
            np.clip(prob_anomaly, 0.0, 1.0, out=prob_anomaly)
            
            # Calculate P(normal) = 1 - P(anomaly)
            prob_normal = 1.0 - prob_anomaly

            # Fill the final array for valid rows
            probabilities[valid_scores_mask, 0] = prob_normal
            probabilities[valid_scores_mask, 1] = prob_anomaly

        # --- Final Validation ---
        # Check if probabilities sum to 1 (within tolerance) for valid rows
        if np.any(valid_scores_mask):
            sums = np.sum(probabilities[valid_scores_mask], axis=1)
            if not np.allclose(sums, 1.0, atol=1e-6):
                warnings.warn("Probabilities do not sum close to 1 for some valid rows after calculation. Check calculations.", RuntimeWarning)

        # Check length consistency between input and final output
        n_samples_in = X_processed_3d.shape[0]
        if n_samples_out != n_samples_in:
            warnings.warn(f"Final probability array length ({n_samples_out}) differs from processed input samples ({n_samples_in}). This might be due to internal model filtering/padding. XAI tools might require matching lengths.", RuntimeWarning)
            # If XAI tools require exact length match, padding/truncation might be needed here.
            if n_samples_out < n_samples_in:
                padded_probs = np.full((n_samples_in, 2), 0.5)
                padded_probs[:n_samples_out] = probabilities  # Fill beginning
                probabilities = padded_probs
            elif n_samples_out > n_samples_in:
                probabilities = probabilities[:n_samples_in]

        print(f"DEBUG predict_proba: Returning final probabilities shape: {probabilities.shape}")
        # Print min/max/mean of final P(anomaly) column
        if probabilities.size > 0:
            print(f"DEBUG predict_proba: Final P(anomaly) stats: min={np.min(probabilities[:, 1]):.4f}, max={np.max(probabilities[:, 1]):.4f}, mean={np.mean(probabilities[:, 1]):.4f}")
        
        return probabilities