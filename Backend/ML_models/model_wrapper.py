import pandas as pd
import numpy as np
from typing import Any, List, Union, Literal # Import Literal
import warnings

class ModelWrapperForXAI:
    """
    Wraps a model with 'detect' and 'get_anomaly_score' methods to provide
    '.predict()' and '.predict_proba()' interfaces for XAI tools.
    Handles different input requirements (DataFrame vs NumPy) of internal model methods
    and different score interpretations (higher vs lower score = anomaly).
    """
    def __init__(self,
                 actual_model_instance: Any,
                 feature_names: List[str],
                 score_interpretation: Literal['lower_is_anomaly', 'higher_is_anomaly'] = 'higher_is_anomaly'
                 ):
        """
        Args:
            actual_model_instance: Trained model instance with 'detect' & 'get_anomaly_score'.
            feature_names (List[str]): Feature names list.
            score_interpretation (str): How to interpret the output of 'get_anomaly_score'.
                                        Set to 'lower_is_anomaly' (e.g., OCSVM) or
                                        'higher_is_anomaly' (e.g., AE reconstruction error).
                                        Defaults to 'higher_is_anomaly'.
        """
        self._model = actual_model_instance
        self._feature_names = feature_names
        self._num_classes = 2
        self._score_interpretation = score_interpretation

        # --- Checks ---
        if not all(hasattr(self._model, meth) and callable(getattr(self._model, meth))
                   for meth in ['detect', 'get_anomaly_score']):
            raise AttributeError("Provided model must have callable 'detect' and 'get_anomaly_score' methods.")
        if len(self._feature_names) == 0: raise ValueError("Feature names list cannot be empty.")
        if self._score_interpretation not in ['lower_is_anomaly', 'higher_is_anomaly']:
            raise ValueError("score_interpretation must be 'lower_is_anomaly' or 'higher_is_anomaly'")
        if not hasattr(self._model, 'sequence_length'):
             warnings.warn("Wrapped model lacks 'sequence_length' attribute.", RuntimeWarning)
        # --- End Checks ---

        print(f"ModelWrapperForXAI initialized for model type: {type(self._model).__name__}")
        print(f"Score interpretation set to: '{self._score_interpretation}'")

    @property
    def sequence_length(self): return getattr(self._model, 'sequence_length', None)

    # --- Private helper to call internal model method ---
    # This handles potential looping ONLY if the underlying model requires it.
    # Assumes the model's methods now handle DataFrame/NumPy input themselves.
    def _call_internal_method(self, X_input_data: Union[pd.DataFrame, np.ndarray], internal_method_name: str) -> np.ndarray:
        """ Calls the specified method on the wrapped model. """
        internal_method = getattr(self._model, internal_method_name)
        # print(f"Wrapper: Calling internal '{internal_method_name}'...") # Optional print
        try:
            results = internal_method(X_input_data)
            return np.array(results) # Ensure NumPy array output
        except Exception as e:
            print(f"ERROR in ModelWrapper calling internal model.{internal_method_name}: {e}")
            # Return array of NaNs matching expected output length
            # For sequence models, output length matches input sequence count
            # For non-sequence models, output length matches input sample count
            n_out = X_input_data.shape[0] if X_input_data.ndim in [2,3] else 1
            return np.full(n_out, np.nan)


    def predict(self, X_np_3d: np.ndarray) -> np.ndarray:
        """ Calls internal model's 'detect', returns 2D array (samples, 1). """
        # LSTM/SVM detect methods should now handle 3D/2D input respectively
        raw_predictions = self._call_internal_method(X_np_3d, 'detect')

        results_np = np.array(raw_predictions, dtype=float)
        bool_mask = ~np.isnan(results_np)
        if pd.api.types.is_bool_dtype(results_np[bool_mask]):
            results_np[bool_mask] = results_np[bool_mask].astype(int)

        if results_np.ndim == 1: results_np = results_np[:, np.newaxis]
        elif results_np.ndim == 0: results_np = np.array([[results_np.item()]])

        # print(f"ModelWrapper.predict: Returning shape {results_np.shape}") # Optional
        return results_np

    def _sigmoid(self, x):
        x_clipped = np.clip(x, -700, 700)
        return 1 / (1 + np.exp(-x_clipped))

    def predict_proba(self, X_np_input: np.ndarray) -> np.ndarray:
        """ Calls internal 'get_anomaly_score', converts scores to probabilities [P(normal), P(anomaly)]. """
        # If input shape is already probability-like (samples, 2) or not the expected 3D shape,
        # handle it gracefully. This suggests DiCE might be calling this directly with internal data.
        expected_ndim = 3 # Expecting (samples, seq_len, features)
        expected_features = len(self._feature_names)
        if X_np_input.ndim != expected_ndim or X_np_input.shape[-1] != expected_features:
            warnings.warn(f"Predict_proba received unexpected input shape {X_np_input.shape}. Expected 3D with {expected_features} features. Returning neutral probabilities.", RuntimeWarning)
            # Return neutral probabilities [0.5, 0.5] for the number of samples received
            num_samples = X_np_input.shape[0] if X_np_input.ndim >= 1 else 1
            return np.full((num_samples, 2), 0.5)
        
        X_np_3d = X_np_input # Rename for clarity if needed, or just use X_np_input
        
        scores_1d = self._call_internal_method(X_np_3d, 'get_anomaly_score')

        scores_clean = np.array(scores_1d, dtype=float)
        print(f"DEBUG predict_proba: Raw scores min/max/mean: {np.nanmin(scores_clean):.4f} / {np.nanmax(scores_clean):.4f} / {np.nanmean(scores_clean):.4f}")
        nan_mask = np.isnan(scores_clean)
        if np.any(nan_mask):
            warnings.warn("NaNs in scores. Probabilities will be [0.5, 0.5].", RuntimeWarning)
            scores_clean[nan_mask] = 0 # Neutral score for NaN

            # Get threshold (assuming model stores it)
        threshold = getattr(self._model, 'threshold', 0.0) # Default threshold to 0 if not found
        if threshold is None: threshold = 0.0 # Handle None case

        # --- Convert scores to P(anomaly) based on interpretation ---
        # You NEED to tune the 'scale_factor'. Start with 1.0 maybe, or std dev of scores?
        scale_factor = 1 # Adjust this based on typical score ranges!
        if self._score_interpretation == 'higher_is_anomaly': # e.g., LSTM AE error
            scaled_scores = (scores_clean - threshold) / scale_factor
            prob_anomaly = self._sigmoid(scaled_scores)
            print(f"predict_proba: Higher score interpretation. Using threshold {threshold:.4f}, scale {scale_factor:.4f}")
        else: # lower_is_anomaly (e.g., SVM score)
            scaled_scores = (scores_clean - threshold) / scale_factor # Note: might need -(...) depending on desired sigmoid direction
            prob_anomaly = self._sigmoid(-scaled_scores) # Lower score relative to threshold -> higher probability
            print(f"predict_proba: Lower score interpretation. Using threshold {threshold:.4f}, scale {scale_factor:.4f}")

        prob_normal = 1.0 - prob_anomaly
        probabilities = np.vstack([prob_normal, prob_anomaly]).T
        print(f"DEBUG predict_proba: Probabilities shape: {probabilities.shape}")

        if not np.allclose(np.sum(probabilities[~nan_mask], axis=1), 1.0):
            warnings.warn("Probabilities do not sum to 1.", RuntimeWarning)

        # print(f"ModelWrapper.predict_proba: Returning shape {probabilities.shape}") # Optional
        return probabilities