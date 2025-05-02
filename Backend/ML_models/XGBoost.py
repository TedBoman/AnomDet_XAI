import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split # Added for validation split
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
        self.best_score: Optional[float] = None # Store best validation score
        self.best_iteration: Optional[int] = None # Store best iteration

        # --- Store configuration from kwargs ---
        self.random_state = kwargs.get('random_state', 42)
        self.validation_set_size = kwargs.get('validation_set_size', 0.15) # Default 15%
        if not 0 < self.validation_set_size < 1:
            raise ValueError("validation_set_size must be between 0 and 1.")

        # Base XGBoost parameters (excluding scale_pos_weight, calculated later)
        self.model_params = {
            'n_estimators': kwargs.get('n_estimators', 100),
            'learning_rate': kwargs.get('learning_rate', 0.1),
            'max_depth': kwargs.get('max_depth', 6),
            'objective': kwargs.get('objective', 'binary:logistic'),
            'eval_metric': kwargs.get('eval_metric', 'logloss'), # Important for early stopping
            'random_state': self.random_state,
            'n_jobs': kwargs.get('n_jobs', -1)
            # Add other relevant base params here if needed
        }
        # Add any other valid kwargs intended for XGBClassifier
        allowed_xgb_params = set(xgb.XGBClassifier().get_params().keys())
        print(allowed_xgb_params)
        # Exclude params managed separately
        managed_params = {'scale_pos_weight', 'early_stopping_rounds'} 
        extra_xgb_params = {
            k: v for k, v in kwargs.items() 
            if k in allowed_xgb_params 
            and k not in self.model_params 
            and k not in managed_params
        }
        self.model_params.update(extra_xgb_params)

        # Store early stopping and verbosity settings
        self.early_stopping_rounds = kwargs.get('early_stopping_rounds', None) # Default off
        self.verbose_eval = kwargs.get('verbose_eval', False) # Default quiet

        # Store calibration method if provided, default to isotonic
        self.calibration_method = kwargs.get('calibration_method', 'isotonic') # 'isotonic' or 'sigmoid'

        print(f"XGBoostModel Initialized with base params: {self.model_params}")
        print(f"Validation split: {self.validation_set_size*100}%, Early Stopping Rounds: {self.early_stopping_rounds}")
        print(f"Calibration Method: {self.calibration_method}, Verbose Eval: {self.verbose_eval}")
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
        Trains the XGBoost classifier with validation split and applies calibration.

        Args:
            X: Input data (DataFrame or 3D NumPy). The *entire* training dataset.
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

        if X_processed_scaled.shape[0] == 0 or y_aligned is None:
            warnings.warn("No data or labels available for training after preprocessing.", RuntimeWarning)
            self.model = None
            return
            
        if X_processed_scaled.shape[0] < 10: # Arbitrary small number
            warnings.warn(f"Very small dataset ({X_processed_scaled.shape[0]} samples), validation split might be ineffective.", RuntimeWarning)
            # Decide if you want to proceed without validation or raise error
            # For now, we proceed but validation might be empty or tiny

        # --- Step 0: Split data into Training and Validation Sets ---
        # Stratify ensures proportion of labels is maintained in train/val splits
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X_processed_scaled,
                y_aligned,
                test_size=self.validation_set_size,
                random_state=self.random_state,
                stratify=y_aligned # Important for imbalanced datasets
            )
            print(f"Data split: Train shape={X_train.shape}, Validation shape={X_val.shape}")
            if X_val.shape[0] == 0:
                warnings.warn("Validation set is empty after split. Disabling early stopping and using training data for calibration.", RuntimeWarning)
                # Fallback: Use training data for calibration, disable early stopping
                X_val, y_val = X_train, y_train # Use train data if val is empty
                effective_early_stopping_rounds = None
                eval_set = None
            else:
                effective_early_stopping_rounds = self.early_stopping_rounds
                eval_set = [(X_val, y_val)] # XGBoost expects a list of tuples

        except ValueError as e:
            warnings.warn(f"Could not stratify split (maybe only one class present?): {e}. Training without validation set.", RuntimeWarning)
            # Fallback: Train on all data, no early stopping, use all data for calibration
            X_train, y_train = X_processed_scaled, y_aligned
            X_val, y_val = X_train, y_train # Use train data for calibration
            effective_early_stopping_rounds = None
            eval_set = None


        # Handle Class Imbalance (using the new TRAINING split labels)
        n_neg_train = np.sum(y_train == 0); n_pos_train = np.sum(y_train == 1)
        scale_pos_weight = 1.0 # Default to float
        if n_pos_train == 0: warnings.warn("No positive samples (label=1) found in the TRAINING split.", RuntimeWarning)
        elif n_neg_train == 0: warnings.warn("No negative samples (label=0) found in the TRAINING split.", RuntimeWarning)
        else: scale_pos_weight = float(n_neg_train) / float(n_pos_train) # Ensure float division
        print(f"Calculated scale_pos_weight based on TRAINING split: {scale_pos_weight:.4f}")

        # --- Use base parameters stored during __init__ and add scale_pos_weight ---
        current_model_params = self.model_params.copy()
        current_model_params['scale_pos_weight'] = scale_pos_weight
        # Add max_delta_step if needed for stability with imbalance
        current_model_params.setdefault('max_delta_step', 1)

        # --- Step 1: Train the Base XGBoost Classifier with Validation ---
        print(f"Training BASE XGBClassifier with {X_train.shape[0]} train samples, {X_val.shape[0]} validation samples...")
        print(f"Using effective XGBoost parameters: {current_model_params}")
        print(f"Early stopping rounds: {effective_early_stopping_rounds}, Verbose: {self.verbose_eval}")
        
        base_xgb_model = xgb.XGBClassifier(**current_model_params)

        fit_params = {}
        if eval_set and effective_early_stopping_rounds is not None and effective_early_stopping_rounds > 0:
            fit_params['early_stopping_rounds'] = effective_early_stopping_rounds
            fit_params['eval_set'] = eval_set
            fit_params['verbose'] = self.verbose_eval
        elif self.verbose_eval: # If verbose is True but no early stopping
            fit_params['eval_set'] = eval_set
            fit_params['verbose'] = self.verbose_eval


        try:
            # Fit using the TRAINING split, validate on the VALIDATION split
            base_xgb_model.fit(X_train, y_train, **fit_params) 
            
            # Store best score and iteration if early stopping was used
            if 'early_stopping_rounds' in fit_params:
                # XGBoost >= 1.6 stores results directly
                if hasattr(base_xgb_model, 'best_score') and hasattr(base_xgb_model, 'best_iteration'):
                    self.best_score = base_xgb_model.best_score
                    self.best_iteration = base_xgb_model.best_iteration
                    print(f"Early stopping triggered. Best iteration: {self.best_iteration}, Best score ({base_xgb_model.eval_metric}): {self.best_score:.4f}")
                else: # Older versions might require accessing results differently, TBD if needed
                    print("Early stopping was enabled, but couldn't retrieve best_score/best_iteration (check XGBoost version compatibility if needed).")


        except Exception as e:
            raise RuntimeError(f"Base XGBoost fitting failed: {e}") from e
        print("Base model training complete.")

        # Optional: Print base model probability stats on VALIDATION set for debugging
        try:
            base_probs_val = base_xgb_model.predict_proba(X_val)
            if base_probs_val.shape[1] > 1:
                print(f"DEBUG: Base P(anomaly) on VAL set stats: min={np.min(base_probs_val[:, 1]):.4f}, max={np.max(base_probs_val[:, 1]):.4f}, mean={np.mean(base_probs_val[:, 1]):.4f}")
        except Exception as prob_e: print(f"Debug predict_proba on validation set failed: {prob_e}")

        # --- Step 2: Apply Probability Calibration using the VALIDATION set ---
        # We use the validation set (X_val, y_val) to calibrate the already trained base model.
        # This avoids calibrating on data the base model was trained on (X_train).
        calibration_method = self.calibration_method # Use stored method
        print(f"Applying probability calibration (method='{calibration_method}') using validation data ({X_val.shape[0]} samples)...")
        
        # Use 'prefit' because base_xgb_model is already trained.
        # The CalibratedClassifierCV's *fit* method will then use the data provided (X_val, y_val)
        # to train the calibrators (isotonic regression or logistic regression).
        calibrated_model = CalibratedClassifierCV(
            estimator=base_xgb_model, # The model trained on X_train
            method=calibration_method,
            cv='prefit' 
        )

        try:
            # Fit the calibrator using the validation data
            calibrated_model.fit(X_val, y_val) 
        except ValueError as e:
             # Catch potential errors if validation set is too small or has only one class
             warnings.warn(f"Probability calibration fitting failed: {e}. The model stored will be the UNCALIBRATED base XGBoost model.", RuntimeWarning)
             self.model = base_xgb_model # Store uncalibrated model as fallback
        except Exception as e:
            raise RuntimeError(f"Probability calibration fitting failed unexpectedly: {e}") from e
        else:
             print("Calibration complete.")
             # --- Step 3: Store the CALIBRATED Model ---
             self.model = calibrated_model # Store the calibrated model
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