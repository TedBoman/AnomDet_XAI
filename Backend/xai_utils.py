# File: xai_utils.py

import numpy as np
import pandas as pd
from typing import Any, Union
import warnings

class SequenceTo2DWrapper:
    """
    Wraps a model expecting 2D input (samples, features) to accept 3D sequence
    input (samples, steps, features) by selecting the LAST time step.

    Provides a `.predict()` method for XAI tools and forwards `sequence_length`.
    """
    def __init__(self, actual_model_instance: Any, sequence_length: int):
        """
        Args:
            actual_model_instance: Trained model instance with a method like
                                   `get_anomaly_score` that expects 2D input.
            sequence_length (int): The sequence length defined for the XAI framework.
        """
        self._model = actual_model_instance
        # Store sequence length - crucial for the framework's assumptions
        self._sequence_length = sequence_length

        # Check for the required score method
        self._score_method_name = 'get_anomaly_score' # Or 'predict_proba' if that exists and returns scores/probs
        if not hasattr(self._model, self._score_method_name) or not callable(getattr(self._model, self._score_method_name)):
             raise AttributeError(f"Provided model instance does not have a callable '{self._score_method_name}' method.")
        print(f"SequenceTo2DWrapper initialized for model type: {type(self._model).__name__}")

    @property
    def sequence_length(self):
        """Exposes sequence length expected by the explainer framework."""
        # This is the sequence length of the *input data* it receives,
        # not necessarily related to the internal model logic.
        return self._sequence_length

    # --- Implement predict() for the explainer ---
    def predict(self, X_np_3d: np.ndarray) -> np.ndarray:
        """
        Accepts 3D sequence input, extracts LAST time step, calls internal
        model's 2D score function, returns scores formatted for SHAP.
        """
        # print(f"SequenceTo2DWrapper.predict: Received input shape {X_np_3d.shape}") # Can be very verbose
        if X_np_3d.ndim != 3:
             raise ValueError(f"Wrapper expects 3D NumPy input (samples, steps, features), got {X_np_3d.ndim}D.")
        # Optional check: compare X_np_3d.shape[1] with self.sequence_length

        # --- Select the last time step -> shape (samples, features) ---
        X_last_step_2d = X_np_3d[:, -1, :]

        # --- Call the original model's scoring function ---
        # Assumes get_anomaly_score handles any needed scaling/encoding internally
        score_func = getattr(self._model, self._score_method_name)
        scores = score_func(X_last_step_2d) # Pass 2D data

        # --- Format output for KernelExplainer (needs 2D: samples, outputs) ---
        if not isinstance(scores, np.ndarray):
             warnings.warn(f"Score function returned non-NumPy type: {type(scores)}. Attempting conversion.", RuntimeWarning)
             scores = np.array(scores)

        if scores.ndim == 1:
             scores = scores[:, np.newaxis] # Reshape (samples,) to (samples, 1)
        elif scores.ndim == 0: # Single sample input case
             scores = np.array([[scores.item()]])
        elif scores.ndim != 2:
             # If score function returns >1 output per sample, adjust or raise error
             raise ValueError(f"Internal model's score function returned unexpected shape: {scores.shape}. Expected 1D or 2D (samples, n_outputs).")

        # print(f"SequenceTo2DWrapper: Returning scores shape {scores.shape}") # Can be verbose
        return scores # Return 2D scores (samples, n_outputs=1)