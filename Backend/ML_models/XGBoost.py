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

    def __init__(self, **kwargs):
        """
        Initializes the XGBoost classifier model using parameters from kwargs.

        Expected kwargs (examples):
            n_estimators (int): Number of boosting rounds (default: 100).
            learning_rate (float): Step size shrinkage (default: 0.1).
            max_depth (int): Maximum depth of a tree (default: 6).
            objective (str): Specifies the learning task (default: 'binary:logistic').
            eval_metric (str): Evaluation metric for validation data (default: 'logloss').
            random_state (int): Random number seed (default: 42).
            n_jobs (int): Number of parallel threads (default: -1).
            scale_pos_weight (float): Controls balance of positive/negative weights (calculated in run).
            ... other XGBClassifier parameters ...
        """
        self.model: Optional[CalibratedClassifierCV] = None # Stores the calibrated model
        self.scaler: Optional[MinMaxScaler] = None
        self.input_type: Optional[str] = None
        self.processed_feature_names: Optional[List[str]] = None
        self.sequence_length: Optional[int] = None
        self.n_original_features: Optional[int] = None
        self.label_col: Optional[str] = None

        # --- Store configuration from kwargs ---
        # Base XGBoost parameters (excluding scale_pos_weight, calculated later)
        self.model_params = {
            'n_estimators': kwargs.get('n_estimators', 100), # Default changed
            'learning_rate': kwargs.get('learning_rate', 0.1),
            'max_depth': kwargs.get('max_depth', 6),
            'objective': kwargs.get('objective', 'binary:logistic'),
            'eval_metric': kwargs.get('eval_metric', 'logloss'),
            'random_state': kwargs.get('random_state', 42),
            'n_jobs': kwargs.get('n_jobs', -1)
            # Add other relevant base params here if needed
        }
        # Add any other valid kwargs intended for XGBClassifier
        allowed_xgb_params = set(xgb.XGBClassifier().get_params().keys())
        extra_xgb_params = {k: v for k, v in kwargs.items() if k in allowed_xgb_params and k not in self.model_params and k != 'scale_pos_weight'}
        self.model_params.update(extra_xgb_params)

        # Store calibration method if provided, default to isotonic
        self.calibration_method = kwargs.get('calibration_method', 'isotonic') # 'isotonic' or 'sigmoid'

        print(f"XGBoostModel Initialized with base params: {self.model_params}, Calibration: {self.calibration_method}")
        # Note: scale_pos_weight is calculated and added during run()

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
            # --- MODIFICATION STARTS HERE ---
            # Handle both 3D and 2D NumPy input, especially for inference
            print(f"DEBUG _prepare_data: Processing NumPy input shape {X.shape}")

            original_ndim = X.ndim
            temp_X = X # Work with a temporary variable

            # If input is 2D during inference, reshape to 3D assuming seq_len=1
            if not is_training and original_ndim == 2:
                n_samples_2d, n_feat_2d = temp_X.shape
                # Check if feature count matches expected original features
                if self.n_original_features is not None and n_feat_2d != self.n_original_features:
                     raise ValueError(f"Inference 2D NumPy input has {n_feat_2d} features, expected {self.n_original_features}.")
                temp_X = temp_X[:, np.newaxis, :] # Reshape to (samples, 1, features)
                print(f"DEBUG _prepare_data: Reshaped 2D NumPy input to 3D {temp_X.shape} for inference.")

            # Now proceed assuming temp_X is 3D (either originally or after reshape)
            if temp_X.ndim != 3:
                 # This check should now primarily catch invalid initial shapes other than 2D during inference
                 raise ValueError(f"NumPy X must be 3D (or 2D during inference), got {original_ndim}D initially.")

            n_samples, seq_len, n_feat = temp_X.shape

            if is_training:
                # Training path expects 3D input initially
                if original_ndim != 3: # Ensure original input was 3D for training
                    raise ValueError(f"NumPy training input must be 3D, got {original_ndim}D.")

                print(f"DEBUG _prepare_data (Train): NumPy input shape {temp_X.shape}") # Add shape print
                self.input_type = 'numpy'
                self.sequence_length = seq_len
                self.n_original_features = n_feat

                if y is None: raise ValueError("Labels 'y' required for NumPy training.")
                if not isinstance(y, np.ndarray) or y.ndim != 1 or len(y) != n_samples:
                    raise ValueError("Invalid 'y' for NumPy training.")

                # Flatten 3D -> 2D for XGBoost training
                X_flattened = temp_X.reshape(n_samples, seq_len * n_feat)
                print(f"DEBUG _prepare_data (Train): Flattened 3D to 2D for scaling/training: {X_flattened.shape}")

                # Scaling (Fit and Transform flattened data)
                self.scaler = MinMaxScaler()
                X_processed_scaled = self.scaler.fit_transform(X_flattened)
                print(f"DEBUG _prepare_data (Train): Scaled 2D data shape: {X_processed_scaled.shape}")

                # Generate flattened feature names
                n_flattened_features = seq_len * n_feat
                self.processed_feature_names = [f"feature_{i}_step_{j}" for j in range(seq_len) for i in range(n_feat)]
                if len(self.processed_feature_names) != n_flattened_features:
                     warnings.warn("Feature name generation mismatch.")
                     self.processed_feature_names = [f"flat_feature_{k}" for k in range(n_flattened_features)]

                y_aligned = y
                # Return 2D scaled data for XGBoost training
                return X_processed_scaled, y_aligned, self.processed_feature_names

            else: # Detection/Inference for NumPy input
                print(f"DEBUG _prepare_data (Inference): NumPy input shape {temp_X.shape}")
                if self.scaler is None or self.sequence_length is None or self.n_original_features is None or self.processed_feature_names is None:
                    raise RuntimeError("Model (trained on NumPy or receiving NumPy for inference) not ready.")

                # Validate sequence length and features against trained values
                # Note: If input was 2D and reshaped, seq_len will be 1 here.
                # This check might need adjustment if models trained on numpy should *only* accept their original seq_len during inference.
                # For now, we allow seq_len=1 if input was 2D.
                if self.input_type == 'numpy' and seq_len != self.sequence_length:
                    # Only strictly enforce seq_len if model was trained on numpy
                     raise ValueError(f"Input seq len {seq_len} != train seq len {self.sequence_length} for NumPy-trained model.")
                if n_feat != self.n_original_features:
                     raise ValueError(f"Input features {n_feat} != train features {self.n_original_features}.")

                # Flatten 3D -> 2D for XGBoost prediction
                X_flattened = temp_X.reshape(n_samples, seq_len * n_feat)
                print(f"DEBUG _prepare_data (Inference): Flattened 3D to 2D for scaling/prediction: {X_flattened.shape}")

                # Scaling (Transform only)
                X_processed_scaled = self.scaler.transform(X_flattened)
                print(f"DEBUG _prepare_data (Inference): Scaled 2D data shape: {X_processed_scaled.shape}")

                # Return 2D scaled data for XGBoost prediction
                return X_processed_scaled, None, self.processed_feature_names # No labels

        else:
             raise TypeError("Input 'X' must be pandas DataFrame or NumPy array.")


    def run(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[np.ndarray] = None, label_col: str = 'label'):
        """
        Trains the XGBoost classifier and applies calibration, using parameters set during __init__.

        Args:
            X: Input data (DataFrame or 3D NumPy).
            y: Target labels (required if X is NumPy array).
            label_col: Name of the target label column (used if X is DataFrame).
        """
        if isinstance(X, pd.DataFrame):
            if y is not None: warnings.warn("Arg 'y' ignored for DataFrame input.", UserWarning)
            print("Running training with DataFrame input (processed as 2D NumPy)...")
            X_processed_scaled, y_aligned, _ = self._prepare_data_for_model(
                X, y=None, label_col=label_col, is_training=True
            )
        elif isinstance(X, np.ndarray):
            print("Running training with 3D NumPy input (flattened for XGBoost)...")
            X_processed_scaled, y_aligned, _ = self._prepare_data_for_model(
                X, y=y, label_col=None, is_training=True
            )
        else:
             raise TypeError("Input 'X' must be pandas DataFrame or 3D NumPy array.")

        if X_processed_scaled.shape[0] == 0:
             warnings.warn("No data for training after preprocessing.", RuntimeWarning)
             self.model = None
             return

        # Handle Class Imbalance
        n_neg = np.sum(y_aligned == 0); n_pos = np.sum(y_aligned == 1)
        scale_pos_weight = 1.0 # Default to float
        if n_pos == 0: warnings.warn("No positive samples (label=1) found in training data.", RuntimeWarning)
        elif n_neg == 0: warnings.warn("No negative samples (label=0) found in training data.", RuntimeWarning)
        else: scale_pos_weight = float(n_neg) / float(n_pos) # Ensure float division
        print(f"Calculated scale_pos_weight: {scale_pos_weight:.4f}")

        # --- Use base parameters stored during __init__ and add scale_pos_weight ---
        current_model_params = self.model_params.copy()
        current_model_params['scale_pos_weight'] = scale_pos_weight
        # Add max_delta_step if needed for stability with imbalance
        current_model_params.setdefault('max_delta_step', 1)

        # --- Step 1: Train the Base XGBoost Classifier ---
        print(f"Training BASE XGBClassifier with {X_processed_scaled.shape[0]} samples, {X_processed_scaled.shape[1]} features...")
        print(f"Using effective XGBoost parameters: {current_model_params}")
        base_xgb_model = xgb.XGBClassifier(**current_model_params)

        try:
            base_xgb_model.fit(X_processed_scaled, y_aligned)
        except Exception as e:
             raise RuntimeError(f"Base XGBoost fitting failed: {e}") from e
        print("Base model training complete.")

        # Optional: Print base model probability stats for debugging
        try:
            base_probs = base_xgb_model.predict_proba(X_processed_scaled)
            if base_probs.shape[1] > 1:
                print(f"DEBUG: Base P(anomaly) stats: min={np.min(base_probs[:, 1]):.4f}, max={np.max(base_probs[:, 1]):.4f}, mean={np.mean(base_probs[:, 1]):.4f}")
        except Exception as prob_e: print(f"Debug predict_proba failed: {prob_e}")

        # --- Step 2: Apply Probability Calibration ---
        calibration_method = self.calibration_method # Use stored method
        print(f"Applying probability calibration (method='{calibration_method}')...")
        calibrated_model = CalibratedClassifierCV(
            estimator=base_xgb_model,
            method=calibration_method,
            cv='prefit' # Base estimator is already fitted
        )

        try:
            calibrated_model.fit(X_processed_scaled, y_aligned)
        except Exception as e:
             raise RuntimeError(f"Probability calibration fitting failed: {e}") from e
        print("Calibration complete.")

        # --- Step 3: Store the CALIBRATED Model ---
        self.model = calibrated_model
        print(f"Stored calibrated model of type: {type(self.model)}")

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