import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.calibration import CalibratedClassifierCV
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

    def __init__(self, n_estimators=350, learning_rate=0.1, max_depth=6, random_state=42, **kwargs):
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
            'eval_metric': 'logloss',
            'random_state': random_state, 'n_jobs': -1
        }
        self.model_params.update(kwargs)
        print(f"XGBoostModel Initialized with base params: {self.model_params}")

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
            print(f"DEBUG _prepare_data: Processing NumPy input shape {X.shape}") # Add shape print
            self.input_type = 'numpy'
            if X.ndim != 3: raise ValueError(f"NumPy X must be 3D (samples, seq_len, features), got {X.ndim}D")
            n_samples, seq_len, n_feat = X.shape
            self.sequence_length = seq_len # Store sequence length
            self.n_original_features = n_feat # Store original feature count

            if is_training:
                if y is None: raise ValueError("Labels 'y' required for NumPy training.")
                if not isinstance(y, np.ndarray) or y.ndim != 1 or len(y) != n_samples:
                    raise ValueError("Invalid 'y' for NumPy training.")

                # --- Preserve 3D Structure during Scaling ---
                # Reshape 3D -> 2D for scaler: (samples, seq_len, features) -> (samples * seq_len, features)
                X_reshaped_for_scaling = X.reshape(-1, n_feat)
                print(f"DEBUG _prepare_data: Reshaped 3D to 2D for scaling: {X_reshaped_for_scaling.shape}")

                # Scaling (Fit and Transform)
                self.scaler = MinMaxScaler()
                X_processed_scaled = self.scaler.fit_transform(X_reshaped_for_scaling)
                print(f"DEBUG _prepare_data: Scaled 2D data shape: {X_processed_scaled.shape}")

                # Feature names: You might still need flattened names for compatibility elsewhere (e.g., DiCE)
                # Or maybe just the original feature names are sufficient if the 3D model uses them?
                # Let's keep flattened for now, but be aware of this potential mismatch in meaning.
                n_flattened_features = seq_len * n_feat
                self.processed_feature_names = [f"feature_{i}_step_{j}" for j in range(seq_len) for i in range(n_feat)]
                if len(self.processed_feature_names) != n_flattened_features:
                    warnings.warn("Feature name generation mismatch.")
                    self.processed_feature_names = [f"flat_feature_{k}" for k in range(n_flattened_features)]

                y_aligned = y
                return X_processed_scaled, y_aligned, self.processed_feature_names

            else: # Detection for NumPy (preserve 3D)
                if self.scaler is None or self.sequence_length is None or self.n_original_features is None or self.processed_feature_names is None:
                    raise RuntimeError("Model (trained on NumPy) not ready.")
                if seq_len != self.sequence_length: raise ValueError(f"Input seq len {seq_len} != train seq len {self.sequence_length}.")
                if n_feat != self.n_original_features: raise ValueError(f"Input features {n_feat} != train features {self.n_original_features}.")

                # --- Preserve 3D Structure during Scaling ---
                # Reshape 3D -> 2D for scaler
                X_reshaped_for_scaling = X.reshape(-1, n_feat)
                print(f"DEBUG _prepare_data: Reshaped 3D to 2D for scaling: {X_reshaped_for_scaling.shape}")

                # Scaling (Transform only)
                scaled_data_2d = self.scaler.transform(X_reshaped_for_scaling)
                print(f"DEBUG _prepare_data: Scaled 2D data shape: {scaled_data_2d.shape}")

                # Return the 3D SCALED array
                return scaled_data_2d, None, self.processed_feature_names # No labels

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
        current_model_params['max_delta_step'] = 1

        # --- Step 1: Train the Base XGBoost Classifier ---
        print(f"Training BASE XGBClassifier with {X_processed_scaled.shape[0]} samples, {X_processed_scaled.shape[1]} features...")
        base_xgb_model = xgb.XGBClassifier(**current_model_params)
        base_xgb_model.fit(X_processed_scaled, y_aligned)
        print("Base model training complete.")

        print("DEBUG: Base XGBoost model predict_proba stats:")
        base_probs = base_xgb_model.predict_proba(X_processed_scaled)
        if base_probs.shape[1] > 1: # Check if it has 2 columns
            print(f"  P(anomaly) min={np.min(base_probs[:, 1]):.4f}, max={np.max(base_probs[:, 1]):.4f}, mean={np.mean(base_probs[:, 1]):.4f}")
        else: # Handle case if predict_proba returns only one column somehow
            print(f"  Probs shape: {base_probs.shape}")
            print(f"  Probs stats: min={np.min(base_probs):.4f}, max={np.max(base_probs):.4f}, mean={np.mean(base_probs):.4f}")


        # --- Step 2: Apply Probability Calibration ---
        print("Applying probability calibration (method='sigmoid')...")
        # We use cv='prefit' because the base_xgb_model is already trained.
        # 'sigmoid' corresponds to Platt scaling. 'isotonic' is another option.
        # NOTE: Ideally, calibration should use a separate validation set,
        # but fitting on the training set is a common practice if one isn't available.
        calibrated_model = CalibratedClassifierCV(
            estimator=base_xgb_model,
            method='isotonic', # Or 'sigmoid'
            cv='prefit'       # Crucial: Indicates the base estimator is already fitted
        )

        # Fit the calibrator
        calibrated_model.fit(X_processed_scaled, y_aligned)
        print("Calibration complete.")

        # --- Step 3: Store the CALIBRATED Model ---
        # Now self.model refers to the calibrated version
        self.model = calibrated_model
        print(f"Stored calibrated model of type: {type(self.model)}")


    def get_anomaly_score(self, detection_data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """ Calculates anomaly scores (probability of class 1). """
        pass


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
    
    def predict_proba(self, X_input: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predicts class probabilities for the input data using the calibrated model.

        Handles DataFrame or 3D NumPy input consistent with training.

        Args:
            X_input (Union[pd.DataFrame, np.ndarray]): Input data for probability prediction.

        Returns:
            np.ndarray: Array of shape (n_samples, 2) with probabilities for class 0 and class 1.
                        Returns empty array shape (0, 2) if no data after processing.
        """
        if self.model is None or self.scaler is None or self.input_type is None or self.processed_feature_names is None or self.n_original_features is None:
            raise RuntimeError("Model is not trained or ready for predict_proba.")

        print(f"Predicting probabilities for input type: {type(X_input)}")
        # Prepare data (returns 2D NumPy array suitable for the model)
        X_processed_scaled, _, _ = self._prepare_data_for_model(
            X_input, is_training=False, label_col=self.label_col
        )

        # Handle case where preprocessing results in no data
        if X_processed_scaled.shape[0] == 0:
             warnings.warn("No data to predict probabilities after preprocessing.", RuntimeWarning)
             return np.empty((0, 2)) # Return shape (0, 2)

        print(f"Input shape to calibrated model's predict_proba: {X_processed_scaled.shape}")
        try:
            # Use the predict_proba method of the CalibratedClassifierCV model
            probabilities = self.model.predict_proba(X_processed_scaled)
        except Exception as e:
            raise RuntimeError(f"Probability prediction failed: {e}") from e

        if probabilities.shape[1] != 2:
             warnings.warn(f"Expected 2 columns in probability output, but got {probabilities.shape[1]}. Returning as is.", RuntimeWarning)

        print(f"Predicted probabilities shape: {probabilities.shape}")
        return probabilities