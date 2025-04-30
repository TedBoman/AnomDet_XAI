import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer 
from ML_models import model_interface 
from typing import List, Optional, Tuple, Union
import warnings

class DecisionTreeModel(model_interface.ModelInterface):
    """
    Supervised Decision Tree classification model for anomaly detection using labeled data.

    Handles input as:
    1.  Pandas DataFrame: Converts features directly to a 2D NumPy array. NO LAGGING.
    2.  3D NumPy array (X) and 1D NumPy array (y): Assumes X has shape
        (samples, sequence_length, features). Flattens the last two dimensions.

    Includes internal scaling (MinMaxScaler) and imputation (SimpleImputer) 
    to handle potential NaN values before feeding data to the Decision Tree.
    
    Trains a DecisionTreeClassifier to predict the anomaly label.
    Handles class imbalance using the 'class_weight' parameter.
    Provides compatibility with SHAP/LIME via the `predict_proba_xai` method.
    """
    
    def __init__(self, *kwargs):
        """Initializes the Decision Tree classifier model.

        Args:
            criterion (str): Function to measure the quality of a split.
            max_depth (int, optional): Maximum depth of the tree.
            min_samples_split (int): Minimum number of samples required to split an internal node.
            min_samples_leaf (int): Minimum number of samples required to be at a leaf node.
            random_state (int): Controls the randomness of the estimator.
            class_weight (dict, 'balanced', optional): Weights associated with classes.
            imputer_strategy (str): Strategy for SimpleImputer ('mean', 'median', 'most_frequent', 'constant').
            **kwargs: Additional parameters passed to DecisionTreeClassifier.
        """
        self.model: Optional[DecisionTreeClassifier] = None
        self.scaler: Optional[MinMaxScaler] = None
        self.imputer: Optional[SimpleImputer] = None
        self.input_type: Optional[str] = None
        self.processed_feature_names: Optional[List[str]] = None
        self.sequence_length: Optional[int] = None
        self.n_original_features: Optional[int] = None
        self.label_col: Optional[str] = None

        # --- Extract parameters from kwargs with defaults ---
        # Core Decision Tree parameters
        criterion = kwargs.get('criterion', 'gini')
        max_depth = kwargs.get('max_depth', None) # Default changed to None
        min_samples_split = kwargs.get('min_samples_split', 2)
        min_samples_leaf = kwargs.get('min_samples_leaf', 1)
        random_state = kwargs.get('random_state', 42)
        class_weight = kwargs.get('class_weight', 'balanced')

        # Imputer strategy
        self._imputer_strategy = kwargs.get('imputer_strategy', 'mean')

        # Store all parameters passed to the underlying model
        self.model_params = {
            'criterion': criterion,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'class_weight': class_weight,
            'random_state': random_state,
        }
        # Add any other kwargs intended for DecisionTreeClassifier
        allowed_dt_params = set(DecisionTreeClassifier().get_params().keys())
        extra_dt_params = {k: v for k, v in kwargs.items() if k in allowed_dt_params and k not in self.model_params}
        self.model_params.update(extra_dt_params)

        if self.model_params.get('class_weight') is None:
            warnings.warn("class_weight was None, setting to 'balanced' by default.", UserWarning)
            self.model_params['class_weight'] = 'balanced'

        print(f"DecisionTreeModel Initialized with params: {self.model_params}, Imputer Strategy: {self._imputer_strategy}")

    def _prepare_data_for_model(
        self, X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        label_col: Optional[str] = None, 
        is_training: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
        """
        Internal helper to preprocess data: reshape (if NumPy), scale, and impute NaNs.

        Returns:
        - X_processed_imputed: Processed, scaled, and imputed features (2D NumPy array).
        - y_aligned: Aligned labels (1D NumPy array, only if y is provided & is_training).
        - feature_names: List of feature names for the columns in X_processed_imputed.
        """
        X_processed_scaled = None
        y_aligned = None
        current_feature_names = None
        
        # --- Stage 1: Reshape (if needed) and get initial features/labels ---
        if isinstance(X, pd.DataFrame):
            if is_training:
                if self.input_type is None: self.input_type = 'dataframe'
                elif self.input_type != 'dataframe': raise RuntimeError("Model already trained with different input type.")
                self.sequence_length = 1 

                if label_col is None or label_col not in X.columns:
                    raise ValueError(f"Label column '{label_col}' not found for DataFrame training.")
                self.label_col = label_col
                original_feature_names = X.columns.drop(label_col).tolist()
                if not original_feature_names: raise ValueError("No feature columns found in DataFrame.")
                
                if self.n_original_features is None: self.n_original_features = len(original_feature_names)
                elif self.n_original_features != len(original_feature_names): raise ValueError("Feature count mismatch during training.")
                
                if self.processed_feature_names is None: self.processed_feature_names = original_feature_names
                elif self.processed_feature_names != original_feature_names: raise ValueError("Feature names mismatch during training.")

                current_feature_names = self.processed_feature_names
                X_features_np = X[current_feature_names].to_numpy()
                y_aligned = X[self.label_col].to_numpy()

                if X_features_np.shape[0] == 0: raise ValueError("No data rows found in DataFrame features.")

            else: # Detection/Scoring for DataFrame
                if self.input_type != 'dataframe' or self.scaler is None or self.imputer is None or self.processed_feature_names is None or self.n_original_features is None:
                    raise RuntimeError("Model was not trained on DataFrame or is not ready.")
                
                missing_cols = set(self.processed_feature_names) - set(X.columns)
                if missing_cols: raise ValueError(f"Detection DataFrame missing required columns: {missing_cols}")

                current_feature_names = self.processed_feature_names
                X_features_np = X[current_feature_names].to_numpy()

                if X_features_np.shape[0] == 0:
                    warnings.warn("No data rows provided for detection/scoring.", RuntimeWarning)
                    return np.empty((0, self.n_original_features)), None, current_feature_names

        elif isinstance(X, np.ndarray):
            if X.ndim != 3: raise ValueError(f"NumPy array input must be 3D (samples, seq_len, features), got {X.ndim}D.")
            n_samples, seq_len, n_feat = X.shape

            if is_training:
                if self.input_type is None: self.input_type = 'numpy'
                elif self.input_type != 'numpy': raise RuntimeError("Model already trained with different input type.")
                
                if self.sequence_length is None: self.sequence_length = seq_len
                elif self.sequence_length != seq_len: raise ValueError("Sequence length mismatch during training.")
                
                if self.n_original_features is None: self.n_original_features = n_feat
                elif self.n_original_features != n_feat: raise ValueError("Feature count mismatch during training.")

                if y is None: raise ValueError("Labels 'y' are required for NumPy array training.")
                if not isinstance(y, np.ndarray) or y.ndim != 1 or len(y) != n_samples:
                    raise ValueError("Invalid 'y' array provided for NumPy training.")
                y_aligned = y

                n_flattened_features = seq_len * n_feat
                X_features_np = X.reshape(n_samples, n_flattened_features)
                
                if self.processed_feature_names is None:
                    self.processed_feature_names = [f"feature_{i}_step_{j}" for j in range(seq_len) for i in range(n_feat)]
                    if len(self.processed_feature_names) != n_flattened_features: # Fallback
                        self.processed_feature_names = [f"flat_feature_{k}" for k in range(n_flattened_features)]
                elif len(self.processed_feature_names) != n_flattened_features:
                    raise ValueError("Flattened feature name count mismatch during training.")
                current_feature_names = self.processed_feature_names

            else: # Detection/Scoring for NumPy
                if seq_len != self.sequence_length: raise ValueError(f"Input sequence length {seq_len} != train sequence length {self.sequence_length}.")
                if n_feat != self.n_original_features: raise ValueError(f"Input feature count {n_feat} != train feature count {self.n_original_features}.")

                current_feature_names = self.processed_feature_names
                if n_samples == 0:
                    warnings.warn("No samples provided for NumPy detection/scoring.", RuntimeWarning)
                    return np.empty((0, len(current_feature_names))), None, current_feature_names

                X_features_np = X.reshape(n_samples, seq_len * n_feat)
        else:
            raise TypeError("Input 'X' must be a pandas DataFrame or a 3D NumPy array.")

        # --- Stage 2: Scaling ---
        if is_training:
            self.scaler = MinMaxScaler()
            X_processed_scaled = self.scaler.fit_transform(X_features_np)
        else:
            if self.scaler is None: raise RuntimeError("Scaler not fitted.")
            X_processed_scaled = self.scaler.transform(X_features_np)
        
        # --- Stage 3: Imputation ---
        if is_training:
            self.imputer = SimpleImputer(strategy=self._imputer_strategy)
            X_processed_imputed = self.imputer.fit_transform(X_processed_scaled)
        else:
            if self.imputer is None: raise RuntimeError("Imputer not fitted.")
            X_processed_imputed = self.imputer.transform(X_processed_scaled)
            
        # Check for remaining NaNs after imputation (should ideally be zero)
        if np.isnan(X_processed_imputed).any():
             warnings.warn("NaN values detected *after* imputation. Check input data or imputer strategy.", RuntimeWarning)
             # Optionally, apply nan_to_num as a final fallback, though fixing the root cause is better
             # X_processed_imputed = np.nan_to_num(X_processed_imputed, nan=0.0) 

        return X_processed_imputed, y_aligned, current_feature_names


    def run(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[np.ndarray] = None, label_col: str = 'label'):
        """
        Trains the Decision Tree classifier after preparing the data (scaling, imputation).

        Args:
            X: Input data (DataFrame or 3D NumPy).
            y: Target labels (required if X is NumPy array). Ignored if X is DataFrame.
            label_col: Name of the target label column (used if X is DataFrame).
        """
        # print(f"Running training for DecisionTreeModel (Input type: {'DataFrame' if isinstance(X, pd.DataFrame) else 'NumPy'})...")
        
        # Prepare data (scaling and imputation happens inside)
        X_processed_imputed, y_aligned, _ = self._prepare_data_for_model(
            X, y=y, label_col=label_col, is_training=True
        )

        if X_processed_imputed.shape[0] == 0:
            warnings.warn("No data available for training after preprocessing.", RuntimeWarning)
            self.model = None
            return

        # Check labels after alignment
        n_neg = np.sum(y_aligned == 0); n_pos = np.sum(y_aligned == 1)
        if n_pos == 0: warnings.warn(f"No positive samples ({self.label_col or 'label'}=1) found in training data.", RuntimeWarning)
        if n_neg == 0: warnings.warn(f"No negative samples ({self.label_col or 'label'}=0) found in training data.", RuntimeWarning)
        
        # Training the Classifier
        # print(f"Training DecisionTreeClassifier with {X_processed_imputed.shape[0]} samples, {X_processed_imputed.shape[1]} features...")
        # print(f"Using model parameters: {self.model_params}")
        self.model = DecisionTreeClassifier(**self.model_params)

        try:
            self.model.fit(X_processed_imputed, y_aligned)
        except Exception as e:
            raise RuntimeError(f"Decision Tree fitting failed: {e}") from e
            
        # print("Model training complete.")
        # if hasattr(self.model, 'classes_'): print(f"Model trained with classes: {self.model.classes_}")


    def get_anomaly_score(self, detection_data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Calculates anomaly scores (probability of class 1) for new data.
        Handles scaling and imputation internally.

        Args:
            detection_data: Input data (DataFrame or 3D NumPy).

        Returns:
            np.ndarray: 1D float array of anomaly scores (probability of class 1), 
                        matching input length. Contains NaN where prediction failed or for empty input.
        """
        if self.model is None or self.scaler is None or self.imputer is None or self.input_type is None or self.processed_feature_names is None:
            raise RuntimeError("Model is not trained or ready for scoring.")

        n_input_samples = len(detection_data) if isinstance(detection_data, pd.DataFrame) else detection_data.shape[0]
        # Initialize scores with NaN, matching original input length
        final_scores = np.full(n_input_samples, np.nan, dtype=float)
        
        if n_input_samples == 0: 
            return final_scores # Return array of NaNs

        try:
            # Prepare data (scaling and imputation happens inside)
            X_processed_imputed, _, _ = self._prepare_data_for_model(
                detection_data, is_training=False, label_col=self.label_col 
            )

            if X_processed_imputed.shape[0] == 0:
                warnings.warn("No processable data found for scoring after preparation.", RuntimeWarning)
                return final_scores # Return array of NaNs

            # Predict Probabilities
            probabilities = self.model.predict_proba(X_processed_imputed)
            
            # Find the index corresponding to the positive class (1)
            if hasattr(self.model, 'classes_'):
                positive_class_index = np.where(self.model.classes_ == 1)[0]
                if len(positive_class_index) > 0:
                    class_index_to_use = positive_class_index[0]
                    anomaly_scores = probabilities[:, class_index_to_use]
                else: 
                    warnings.warn("Positive class (1) not found in model classes during scoring. Returning probability of first class.", RuntimeWarning)
                    class_index_to_use = 0 
                    anomaly_scores = probabilities[:, class_index_to_use] if probabilities.shape[1] > 0 else np.zeros(probabilities.shape[0])
            else:
                warnings.warn("Model classes_ attribute not found. Assuming class 1 is the second column for probabilities.", RuntimeWarning)
                if probabilities.shape[1] < 2:
                     raise RuntimeError("Cannot determine anomaly score: predict_proba returned fewer than 2 columns and classes_ attribute is missing.")
                anomaly_scores = probabilities[:, 1] # Fallback assumption

            # Assign scores - length should match processed data length
            if len(anomaly_scores) == X_processed_imputed.shape[0]:
                 # If processed length matches input length (usual case unless input was empty)
                 if len(anomaly_scores) == n_input_samples:
                     final_scores = anomaly_scores
                 else: # Should only happen if X_processed_imputed became empty when n_input_samples > 0
                      len_to_copy = min(len(anomaly_scores), n_input_samples) 
                      final_scores[:len_to_copy] = anomaly_scores[:len_to_copy]
            else:
                 # This indicates an unexpected internal mismatch, pad just in case
                 warnings.warn(f"Internal mismatch: Score length ({len(anomaly_scores)}) != processed data length ({X_processed_imputed.shape[0]}). Padding.", RuntimeWarning)
                 len_to_copy = min(len(anomaly_scores), n_input_samples) 
                 final_scores[:len_to_copy] = anomaly_scores[:len_to_copy]

        except Exception as e:
            warnings.warn(f"Anomaly score calculation failed: {e}. Returning NaNs for affected samples.", RuntimeWarning)
            # final_scores is already initialized with NaNs, so we just return it
            
        return final_scores


    def detect(self, detection_data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Detects anomalies (predicts class 1) in new data.
        Handles scaling and imputation internally.

        Args:
            detection_data: Input data (DataFrame or 3D NumPy).

        Returns:
            np.ndarray: 1D bool array indicating anomalies (True for class 1), 
                        matching input length. False where prediction failed or for empty input.
        """
        if self.model is None or self.scaler is None or self.imputer is None or self.input_type is None or self.processed_feature_names is None:
            raise RuntimeError("Model is not trained or ready for detection.")

        n_input_samples = len(detection_data) if isinstance(detection_data, pd.DataFrame) else detection_data.shape[0]
        # Initialize detections with False, matching original input length
        final_anomalies = np.zeros(n_input_samples, dtype=bool)

        if n_input_samples == 0: 
            return final_anomalies # Return array of False

        try:
            # Prepare data (scaling and imputation happens inside)
            X_processed_imputed, _, _ = self._prepare_data_for_model(
                detection_data, is_training=False, label_col=self.label_col
            )

            if X_processed_imputed.shape[0] == 0:
                warnings.warn("No processable data found for detection after preparation.", RuntimeWarning)
                return final_anomalies # Return array of False

            # Prediction
            predictions = self.model.predict(X_processed_imputed)
            anomalies = (predictions == 1)

            # Assign results - length should match processed data length
            if len(anomalies) == X_processed_imputed.shape[0]:
                # If processed length matches input length (usual case unless input was empty)
                if len(anomalies) == n_input_samples:
                    final_anomalies = anomalies
                else: # Should only happen if X_processed_imputed became empty when n_input_samples > 0
                    len_to_copy = min(len(anomalies), n_input_samples)
                    final_anomalies[:len_to_copy] = anomalies[:len_to_copy]
            else:
                # This indicates an unexpected internal mismatch, pad just in case
                 warnings.warn(f"Internal mismatch: Detection length ({len(anomalies)}) != processed data length ({X_processed_imputed.shape[0]}). Padding.", RuntimeWarning)
                 len_to_copy = min(len(anomalies), n_input_samples) 
                 final_anomalies[:len_to_copy] = anomalies[:len_to_copy]
                 
        except Exception as e:
            warnings.warn(f"Anomaly detection failed: {e}. Returning False for affected samples.", RuntimeWarning)
            # final_anomalies is already initialized with False

        # num_anomalies = final_anomalies.sum()
        # print(f"Detection complete. Found {num_anomalies} anomalies.")
        return final_anomalies


    # --- METHOD FOR XAI (SHAP/LIME) ---
    def predict_proba(self, X_xai: np.ndarray) -> np.ndarray:
        """
        Prediction function for XAI methods (SHAP/LIME).
        Accepts input in the *original* format expected during training and
        performs internal scaling and imputation.

        Args:
            X_xai: Input data in the original format.
                - If trained on DataFrame: MUST be 2D NumPy (n_instances, n_original_features).
                - If trained on 3D NumPy: MUST be 3D NumPy (n_instances, seq_len, n_original_features).

        Returns:
            np.ndarray: Predicted probabilities with shape (n_instances, n_classes).
        """
        if self.model is None or self.scaler is None or self.imputer is None or self.input_type is None \
           or self.n_original_features is None:
            raise RuntimeError("Model is not trained or ready for XAI prediction.")
        
        if not isinstance(X_xai, np.ndarray):
            raise TypeError("Input X_xai for XAI must be a NumPy array.")

        n_instances = X_xai.shape[0]
        if n_instances == 0:
            n_classes = len(self.model.classes_) if hasattr(self.model, 'classes_') and self.model.classes_ is not None else 2
            return np.empty((0, n_classes))

        X_scaled = None
        X_imputed = None

        # --- Handle based on how the model was trained ---
        try:
            if self.input_type == 'dataframe':
                if X_xai.ndim != 2:
                    raise ValueError(f"XAI input must be 2D (n_instances, n_original_features) for DataFrame-trained model, got {X_xai.ndim}D.")
                if X_xai.shape[1] != self.n_original_features:
                    raise ValueError(f"XAI input has {X_xai.shape[1]} features, expected {self.n_original_features} original features.")

                if not hasattr(self.scaler, 'scale_') or self.scaler.scale_ is None: 
                    raise RuntimeError("Scaler has not been fitted.")
                X_scaled = self.scaler.transform(X_xai)
                
                if not hasattr(self.imputer, 'statistics_') : # Check if imputer is fitted
                     raise RuntimeError("Imputer has not been fitted.")
                X_imputed = self.imputer.transform(X_scaled)

            elif self.input_type == 'numpy':
                if self.sequence_length is None: raise RuntimeError("Sequence length not set for NumPy-trained model.")
                if X_xai.ndim != 3:
                    raise ValueError(f"XAI input must be 3D (n_instances, seq_len, features) for NumPy-trained model, got {X_xai.ndim}D.")

                _, seq_len, n_feat = X_xai.shape
                if seq_len != self.sequence_length: raise ValueError(f"XAI input seq len ({seq_len}) != train seq len ({self.sequence_length}).")
                if n_feat != self.n_original_features: raise ValueError(f"XAI input features ({n_feat}) != train features ({self.n_original_features}).")

                X_reshaped = X_xai.reshape(n_instances, seq_len * n_feat)

                if not hasattr(self.scaler, 'scale_') or self.scaler.scale_ is None: 
                    raise RuntimeError("Scaler has not been fitted.")
                X_scaled = self.scaler.transform(X_reshaped)

                if not hasattr(self.imputer, 'statistics_') : # Check if imputer is fitted
                     raise RuntimeError("Imputer has not been fitted.")
                X_imputed = self.imputer.transform(X_scaled)
                
            else:
                raise RuntimeError(f"Unsupported training input_type '{self.input_type}' for XAI.")

            # --- Predict probabilities using the internal model ---
            if X_imputed is None: 
                raise RuntimeError("Internal error: Imputed data for XAI prediction is None.")
            
            # Final check for NaNs before prediction in XAI context
            if np.isnan(X_imputed).any():
                 warnings.warn("NaNs detected in XAI input *after* imputation. Check input or imputer.", RuntimeWarning)
                 X_imputed = np.nan_to_num(X_imputed, nan=0.0) # Fallback for XAI

            probabilities = self.model.predict_proba(X_imputed)

        except Exception as e:
            # Catch errors during preprocessing or prediction within XAI
            raise RuntimeError(f"XAI prediction failed during preprocessing or model prediction. Error: {e}") from e

        return probabilities