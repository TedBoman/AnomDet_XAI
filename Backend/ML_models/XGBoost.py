import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from ML_models import model_interface 
from typing import List, Dict, Optional, Tuple, Union
import warnings

class XGBoostModel(model_interface.ModelInterface):
    """
    Supervised XGBoost classification model for anomaly detection using labeled data.

    Handles input as:
    1.  Pandas DataFrame: Converts features directly to a 2D NumPy array.
        NOTE: This approach treats each row independently and DOES NOT use
        lagging or time-series context. The `time_steps` parameter is ignored.
        Internal preprocessing for XAI methods IS supported for this input type.
    2.  3D NumPy array (X) and 1D NumPy array (y): Assumes X has shape
        (samples, sequence_length, features). Flattens the last two dimensions.
        Internal preprocessing for XAI methods IS supported for this input type.

    Trains an XGBClassifier to predict the anomaly label.
    Handles class imbalance using 'scale_pos_weight'.
    Provides compatibility with SHAP/LIME via the `predict_proba_xai` method,
    which handles preprocessing internally based on the training input type.
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, **kwargs):
        """
        Initializes the XGBoost classifier model.
        """
        self.model: Optional[xgb.XGBClassifier] = None
        self.scaler: Optional[MinMaxScaler] = None
        self.input_type: Optional[str] = None # 'dataframe' or 'numpy'
        # Feature names AFTER potential processing (lagging removed for DF)
        self.processed_feature_names: Optional[List[str]] = None
        # Sequence length (NumPy) or 1 (DataFrame 2D)
        self.sequence_length: Optional[int] = None
        # Number of original features before any processing
        self.n_original_features: Optional[int] = None
        self.label_col: Optional[str] = None # DataFrame specific

        self.model_params = {
            'n_estimators': n_estimators, 'learning_rate': learning_rate,
            'max_depth': max_depth, 'objective': 'binary:logistic',
            'eval_metric': 'logloss', 'use_label_encoder': False,
            'random_state': random_state, 'n_jobs': -1
        }
        self.model_params.update(kwargs)
        print(f"XGBoostModel Initialized with base params: {self.model_params}")


    # REMOVED: _create_lagged_features (no longer used)

    def _prepare_data_for_model(
        self, X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        # time_steps is now ignored for DataFrame input
        time_steps: Optional[int] = None,
        label_col: Optional[str] = None,
        is_training: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
        """
        Internal helper to preprocess data.
        For DataFrames: Converts features to 2D NumPy, scales.
        For NumPy: Reshapes 3D to 2D, scales.

        Returns:
        - X_processed_scaled: Processed and scaled features (2D NumPy array).
        - y_aligned: Aligned labels (1D NumPy array, only if y is provided).
        - feature_names: List of feature names for the columns in X_processed_scaled.
        """
        if isinstance(X, pd.DataFrame):
            print("Processing DataFrame input as 2D NumPy (no lagging)...")
            self.input_type = 'dataframe'
            self.sequence_length = 1 # No sequence dimension

            if is_training:
                if label_col is None or label_col not in X.columns:
                    raise ValueError(f"Label column '{label_col}' not found.")
                self.label_col = label_col
                original_feature_names = X.columns.drop(label_col).tolist()
                if not original_feature_names: raise ValueError("No feature columns found.")
                self.n_original_features = len(original_feature_names)
                # Processed names are just the original names
                self.processed_feature_names = original_feature_names

                X_features_df = X[original_feature_names]
                y_series = X[self.label_col]

                # Convert features directly to 2D NumPy array
                X_features_np = X_features_df.to_numpy()
                y_aligned = y_series.to_numpy()

                if X_features_np.shape[0] == 0: raise ValueError("No data after feature extraction.")

                # Scaling
                self.scaler = MinMaxScaler()
                X_processed_scaled = self.scaler.fit_transform(X_features_np)

                return X_processed_scaled, y_aligned, self.processed_feature_names

            else: # Detection for DataFrame (convert to 2D NumPy)
                if self.scaler is None or self.processed_feature_names is None or self.n_original_features is None:
                     raise RuntimeError("Model (trained on DataFrame) not ready.")
                missing_cols = set(self.processed_feature_names) - set(X.columns)
                if missing_cols: raise ValueError(f"Missing required columns: {missing_cols}")

                X_features_df = X[self.processed_feature_names]
                X_features_np = X_features_df.to_numpy()

                if X_features_np.shape[0] == 0:
                    warnings.warn("No data provided for detection.", RuntimeWarning)
                    return np.empty((0, self.n_original_features)), None, self.processed_feature_names

                # Scaling
                X_processed_scaled = self.scaler.transform(X_features_np)
                return X_processed_scaled, None, self.processed_feature_names # No labels

        elif isinstance(X, np.ndarray):
            # print("Processing 3D NumPy input...") # Less verbose
            self.input_type = 'numpy'
            if X.ndim != 3: raise ValueError(f"NumPy X must be 3D, got {X.ndim}")
            n_samples, seq_len, n_feat = X.shape
            self.sequence_length = seq_len # Store sequence length
            self.n_original_features = n_feat # Store original feature count

            if is_training:
                if y is None: raise ValueError("Labels 'y' required for NumPy training.")
                if not isinstance(y, np.ndarray) or y.ndim != 1 or len(y) != n_samples:
                     raise ValueError("Invalid 'y' for NumPy training.")

                # Reshape 3D -> 2D
                n_flattened_features = seq_len * n_feat
                X_reshaped = X.reshape(n_samples, n_flattened_features)
                # Generate flattened feature names
                self.processed_feature_names = [f"feature_{i}_step_{j}" for j in range(seq_len) for i in range(n_feat)]
                if len(self.processed_feature_names) != n_flattened_features:
                    warnings.warn("Feature name generation mismatch.")
                    self.processed_feature_names = [f"flat_feature_{k}" for k in range(n_flattened_features)]

                # Scaling
                self.scaler = MinMaxScaler()
                X_processed_scaled = self.scaler.fit_transform(X_reshaped)
                y_aligned = y
                return X_processed_scaled, y_aligned, self.processed_feature_names

            else: # Detection for NumPy (reshape 3D -> 2D)
                if self.scaler is None or self.sequence_length is None or self.n_original_features is None or self.processed_feature_names is None:
                    raise RuntimeError("Model (trained on NumPy) not ready.")
                if seq_len != self.sequence_length: raise ValueError(f"Input seq len {seq_len} != train seq len {self.sequence_length}.")
                if n_feat != self.n_original_features: raise ValueError(f"Input features {n_feat} != train features {self.n_original_features}.")

                X_reshaped = X.reshape(n_samples, seq_len * n_feat)
                X_processed_scaled = self.scaler.transform(X_reshaped)
                return X_processed_scaled, None, self.processed_feature_names # No labels
        else:
            raise TypeError("Input must be pandas DataFrame or 3D NumPy array.")


    # time_steps parameter removed from signature as it's ignored for DataFrames now
    def run(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[np.ndarray] = None, label_col: str = 'label'):
        """
        Trains the XGBoost classifier.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Input data.
                - DataFrame: Features + label column. Treated as 2D tabular data (no lagging).
                - 3D NumPy: Features array (samples, sequence_length, features).
            y (Optional[np.ndarray]): Target labels (required if X is NumPy array).
            label_col (str): Name of the target label column (used if X is DataFrame). Defaults to 'label'.
        """
        if isinstance(X, pd.DataFrame):
            if y is not None: warnings.warn("Arg 'y' ignored for DataFrame input.", UserWarning)
            print("Running training with DataFrame input (processed as 2D NumPy)...")
            # Pass None for time_steps as it's ignored
            X_processed_scaled, y_aligned, _ = self._prepare_data_for_model(
                X, y=None, time_steps=None, label_col=label_col, is_training=True
            )
        elif isinstance(X, np.ndarray):
            print("Running training with 3D NumPy input...")
            X_processed_scaled, y_aligned, _ = self._prepare_data_for_model(
                X, y=y, time_steps=None, label_col=None, is_training=True
            )
        else:
             raise TypeError("Input 'X' must be pandas DataFrame or 3D NumPy array.")

        if X_processed_scaled.shape[0] == 0:
             warnings.warn("No data for training after preprocessing.", RuntimeWarning)
             self.model = None
             return

        # Handle Class Imbalance
        n_neg = np.sum(y_aligned == 0); n_pos = np.sum(y_aligned == 1)
        scale_pos_weight = 1
        if n_pos == 0: warnings.warn("No positive samples found.", RuntimeWarning)
        elif n_neg == 0: warnings.warn("No negative samples found.", RuntimeWarning)
        else: scale_pos_weight = n_neg / n_pos
        print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")
        current_model_params = self.model_params.copy()
        current_model_params['scale_pos_weight'] = scale_pos_weight

        # Training the Classifier
        print(f"Training XGBClassifier with {X_processed_scaled.shape[0]} samples, {X_processed_scaled.shape[1]} features...")
        self.model = xgb.XGBClassifier(**current_model_params)
        # Fit directly on the 2D NumPy array
        self.model.fit(X_processed_scaled, y_aligned)
        print("Model training complete.")


    def get_anomaly_score(self, detection_data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """ Calculates anomaly scores (probability of class 1). """
        if self.model is None or self.scaler is None or self.input_type is None or self.processed_feature_names is None:
             raise RuntimeError("Model is not trained or ready.")

        n_input_samples = len(detection_data) if isinstance(detection_data, pd.DataFrame) else detection_data.shape[0]
        if n_input_samples == 0: return np.array([], dtype=float)

        # Prepare data (returns 2D NumPy array)
        X_processed_scaled, _, _ = self._prepare_data_for_model(
            detection_data, is_training=False, label_col=self.label_col # Pass label_col if needed for column dropping
        )

        if X_processed_scaled.shape[0] == 0:
            # No data remained after potential filtering/validation inside prepare
            return np.full(n_input_samples, np.nan, dtype=float)

        # Predict Probabilities
        try:
            probabilities = self.model.predict_proba(X_processed_scaled)
            positive_class_index = np.where(self.model.classes_ == 1)[0]
            if len(positive_class_index) == 0:
                 anomaly_scores = np.zeros(probabilities.shape[0]) if 0 in self.model.classes_ else np.full(probabilities.shape[0], np.nan)
                 if np.isnan(anomaly_scores).any(): warnings.warn("Positive class (1) not found.", RuntimeWarning)
            else:
                 anomaly_scores = probabilities[:, positive_class_index[0]]
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}") from e

        # Result directly corresponds to input order now (no lagging reindexing)
        if len(anomaly_scores) != n_input_samples:
             warnings.warn(f"Output score length ({len(anomaly_scores)}) mismatch vs input ({n_input_samples}).", RuntimeWarning)
             # Pad with NaN if shorter? Or return as is? Let's pad.
             final_scores = np.full(n_input_samples, np.nan)
             len_to_copy = min(len(anomaly_scores), n_input_samples)
             final_scores[:len_to_copy] = anomaly_scores[:len_to_copy]
             return final_scores

        return anomaly_scores


    def detect(self, detection_data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """ Detects anomalies (predicts class 1). """
        if self.model is None or self.scaler is None or self.input_type is None or self.processed_feature_names is None:
             raise RuntimeError("Model is not trained or ready.")

        n_input_samples = len(detection_data) if isinstance(detection_data, pd.DataFrame) else detection_data.shape[0]
        if n_input_samples == 0: return np.array([], dtype=bool)

        # Prepare data (returns 2D NumPy array)
        X_processed_scaled, _, _ = self._prepare_data_for_model(
            detection_data, is_training=False, label_col=self.label_col
        )

        if X_processed_scaled.shape[0] == 0:
            return np.zeros(n_input_samples, dtype=bool) # Return all False

        # Prediction
        try:
            predictions = self.model.predict(X_processed_scaled)
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}") from e
        anomalies = (predictions == 1)

        # Result directly corresponds to input order
        if len(anomalies) != n_input_samples:
             warnings.warn(f"Output detection length ({len(anomalies)}) mismatch vs input ({n_input_samples}).", RuntimeWarning)
             # Pad with False if shorter?
             final_anomalies = np.zeros(n_input_samples, dtype=bool)
             len_to_copy = min(len(anomalies), n_input_samples)
             final_anomalies[:len_to_copy] = anomalies[:len_to_copy]
             return final_anomalies

        return anomalies


    # --- METHOD FOR XAI (SHAP/LIME) ---
    def predict_proba_xai(self, X_xai: np.ndarray) -> np.ndarray:
        """
        Prediction function for XAI methods (SHAP/LIME) supporting internal
        preprocessing for models trained on EITHER DataFrame (as 2D) OR 3D NumPy arrays.

        Accepts input in the *original* format and performs internal scaling
        (for DataFrame input) or reshaping + scaling (for 3D NumPy input).

        Args:
            X_xai (np.ndarray): Input data in the original format.
                - If trained on DataFrame: MUST be 2D NumPy (n_instances, n_original_features).
                - If trained on 3D NumPy: MUST be 3D NumPy (n_instances, seq_len, n_original_features).

        Returns:
            np.ndarray: Predicted probabilities with shape (n_instances, n_classes).

        Raises:
            RuntimeError: If the model is not trained or prerequisites are missing.
            TypeError: If input type is incorrect for the training mode.
            ValueError: If input dimensions do not match training data dimensions.
        """
        if self.model is None or self.scaler is None or self.input_type is None \
           or self.n_original_features is None: # Use n_original_features for checks
            raise RuntimeError("Model is not trained or ready for XAI prediction.")

        if not isinstance(X_xai, np.ndarray):
            raise TypeError("Input X_xai must be a NumPy array.")

        n_instances = X_xai.shape[0]
        if n_instances == 0:
             # Determine expected number of classes for empty output shape
             n_classes = len(self.model.classes_) if hasattr(self.model, 'classes_') and self.model.classes_ is not None else 2
             return np.empty((0, n_classes))

        X_scaled = None

        # --- Handle based on how the model was trained ---
        if self.input_type == 'dataframe':
            # Expect 2D NumPy input matching original features
            if X_xai.ndim != 2:
                raise ValueError(f"Input X_xai must be 2D (n_instances, n_original_features) for DataFrame-trained model, got {X_xai.ndim}D.")
            if X_xai.shape[1] != self.n_original_features:
                raise ValueError(f"Input X_xai has {X_xai.shape[1]} features, expected {self.n_original_features} original features.")

            # Scale the 2D input
            try:
                if not hasattr(self.scaler, 'scale_'): raise RuntimeError("Scaler not fitted.")
                X_scaled = self.scaler.transform(X_xai)
            except Exception as e:
                raise RuntimeError(f"Failed to scale 2D input for XAI. Shape: {X_xai.shape}. Error: {e}") from e

        elif self.input_type == 'numpy':
             # Expect 3D NumPy input matching original sequence structure
            if X_xai.ndim != 3:
                raise ValueError(f"Input X_xai must be 3D (n_instances, seq_len, features) for NumPy-trained model, got {X_xai.ndim}D.")

            _, seq_len, n_feat = X_xai.shape
            if seq_len != self.sequence_length: raise ValueError(f"Input X_xai seq len ({seq_len}) != train seq len ({self.sequence_length}).")
            if n_feat != self.n_original_features: raise ValueError(f"Input X_xai features ({n_feat}) != train features ({self.n_original_features}).")

            # Reshape 3D -> 2D
            X_reshaped = X_xai.reshape(n_instances, seq_len * n_feat)

            # Scale the reshaped input
            try:
                if not hasattr(self.scaler, 'scale_'): raise RuntimeError("Scaler not fitted.")
                X_scaled = self.scaler.transform(X_reshaped)
            except Exception as e:
                raise RuntimeError(f"Failed to scale reshaped 3D input for XAI. Shape: {X_reshaped.shape}. Error: {e}") from e
        else:
             raise RuntimeError(f"Unsupported input_type '{self.input_type}' for XAI.")

        # --- Predict probabilities using the internal model ---
        try:
            probabilities = self.model.predict_proba(X_scaled)
        except Exception as e:
            raise RuntimeError(f"Internal model prediction failed during XAI call. Scaled input shape: {X_scaled.shape}. Error: {e}") from e

        return probabilities