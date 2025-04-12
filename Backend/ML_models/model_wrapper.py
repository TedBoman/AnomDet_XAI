# In the file where ModelWrapperForXAI is defined

import pandas as pd
import numpy as np
from typing import Any, List, Union
import warnings

class ModelWrapperForXAI:
    """
    Wraps a model with a 'detect' method (expecting DataFrame input)
    to provide the standard 'predict' interface (accepting NumPy)
    expected by XAI tools.
    """
    def __init__(self, actual_model_instance: Any, feature_names: List[str]):
        """
        Args:
            actual_model_instance: The trained model instance (e.g., your LSTM)
                                   that has a 'detect' method expecting a DataFrame.
            feature_names (List[str]): The list of feature names corresponding to the
                                       last dimension of the NumPy arrays passed to predict.
        """
        self._model = actual_model_instance
        self._feature_names = feature_names # Store feature names for DataFrame creation

        if not hasattr(self._model, 'detect') or not callable(self._model.detect):
            raise AttributeError("The provided model instance does not have a callable 'detect' method.")
        if len(self._feature_names) == 0:
             raise ValueError("Feature names list cannot be empty for DataFrame conversion.")
        # Optional: Check for sequence_length if needed elsewhere
        if not hasattr(self._model, 'sequence_length'):
             warnings.warn("Wrapped model lacks 'sequence_length'. Ensure it's handled elsewhere if needed.", RuntimeWarning)

        print(f"ModelWrapperForXAI initialized for model type: {type(self._model).__name__} with {len(self._feature_names)} features.")

    @property
    def sequence_length(self):
        # Forward sequence_length if explainer or other parts need it
        return getattr(self._model, 'sequence_length', None)

    def predict(self, X_np_3d: np.ndarray) -> np.ndarray:
        """
        Accepts a 3D NumPy array (samples, steps, features), converts each
        sequence to a DataFrame, calls the internal model's 'detect' method,
        and returns results as a NumPy array.

        NOTE: Looping can be slow if the internal 'detect' method does not
              support batch predictions on list of DataFrames or 3D NumPy.
        """
        print(f"ModelWrapper: Received NumPy input shape {X_np_3d.shape}. Converting to DataFrame(s) for detect()...")
        if X_np_3d.ndim != 3:
            raise ValueError(f"ModelWrapper expects 3D NumPy input (samples, steps, features), got {X_np_3d.ndim}D.")
        if X_np_3d.shape[2] != len(self._feature_names):
            raise ValueError(
                f"ModelWrapper input feature dimension ({X_np_3d.shape[2]}) doesn't match "
                f"provided feature_names count ({len(self._feature_names)})."
            )

        num_samples = X_np_3d.shape[0]
        predictions = []

        # --- Loop through samples ---
        print(f"ModelWrapper: Looping through {num_samples} samples for prediction...")
        for i in range(num_samples):
            # Convert the i-th sequence (steps, features) to a DataFrame
            # Using stored feature names
            sample_df = pd.DataFrame(X_np_3d[i], columns=self._feature_names)
            try:
                # Call the original model's detect method with the DataFrame
                sample_pred = self._model.detect(sample_df)
                # Handle potential scalar/single-element array output from detect
                if np.isscalar(sample_pred):
                    predictions.append(sample_pred)
                elif isinstance(sample_pred, (np.ndarray, pd.Series, list)) and len(sample_pred) == 1:
                    predictions.append(sample_pred[0] if isinstance(sample_pred, list) else sample_pred.item(0))
                else:
                    # If detect returns >1 value for a single sequence, that's unusual
                    warnings.warn(f"ModelWrapper: detect() returned unexpected format/length for sample {i}: {type(sample_pred)}. Using first element if possible.", RuntimeWarning)
                    try:
                         predictions.append(sample_pred[0] if isinstance(sample_pred, list) else sample_pred.item(0))
                    except:
                         predictions.append(np.nan) # Add NaN if conversion fails
            except Exception as e:
                 print(f"ERROR in ModelWrapper calling internal model.detect for sample {i}: {e}")
                 # Decide how to handle: raise error, or append NaN/default? Let's append NaN.
                 predictions.append(np.nan) # Add NaN on error

        # --- Combine results ---
        print(f"ModelWrapper: Finished prediction loop.")
        # Handle potential NaNs introduced by errors before converting type
        results_np = np.array(predictions)

        # Convert boolean to int if necessary for SHAP (AFTER handling NaNs if applicable)
        if pd.api.types.is_bool_dtype(results_np[~np.isnan(results_np)]): # Check non-NaN elements
             print("ModelWrapper: Converting boolean results to int (NaNs remain NaN).")
             # Convert only non-NaN values to int, keep NaNs as float
             results_np = np.where(np.isnan(results_np), np.nan, results_np.astype(float)) # Convert bools to 1.0/0.0 first
        elif pd.api.types.is_numeric_dtype(results_np[~np.isnan(results_np)]):
             # Ensure float type if numeric (SHAP prefers floats often)
              results_np = results_np.astype(float)


        # Ensure output is 2D (samples, 1 output) for KernelExplainer
        if results_np.ndim == 1:
            results_np = results_np[:, np.newaxis]
        elif results_np.ndim == 0: # Single sample case
            results_np = np.array([[results_np.item()]])

        print(f"ModelWrapper: Returning predictions with shape {results_np.shape}")
        return results_np