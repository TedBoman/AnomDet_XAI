import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer 
from sklearn.model_selection import train_test_split # Added for validation split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score # Added for evaluation
from ML_models import model_interface 
from typing import Dict, List, Optional, Tuple, Union
import warnings
import traceback # Added for better error printing

class DecisionTreeModel(model_interface.ModelInterface): 
    """
    Supervised Decision Tree classification model for anomaly detection using labeled data,
    with internal validation split for performance estimation.

    Handles input as:
    1.  Pandas DataFrame: Converts features directly to a 2D NumPy array. NO LAGGING.
    2.  3D NumPy array (X) and 1D NumPy array (y): Assumes X has shape
        (samples, sequence_length, features). Flattens the last two dimensions.

    Includes internal scaling (MinMaxScaler) and imputation (SimpleImputer).
    Trains a DecisionTreeClassifier on a subset of the training data and evaluates 
    on a held-out validation subset.
    Handles class imbalance using the 'class_weight' parameter.
    """
    
    def __init__(self, **kwargs):
        """Initializes the Decision Tree classifier model.

        Args:
            criterion (str): Function to measure the quality of a split (default: 'gini').
            max_depth (int, optional): Maximum depth of the tree (default: None).
            min_samples_split (int): Minimum samples to split node (default: 2).
            min_samples_leaf (int): Minimum samples at leaf node (default: 1).
            random_state (int): Controls randomness (default: 42).
            class_weight (dict, 'balanced', optional): Class weights (default: 'balanced').
            imputer_strategy (str): Strategy for SimpleImputer ('mean', 'median', etc., default: 'mean').
            validation_set_size (float): Proportion of training data for validation (default: 0.15).
            validation_metrics (list):  Metrics to compute on validation set 
                                        (e.g., ['accuracy', 'f1', 'roc_auc'], default: ['accuracy', 'f1']).
            **kwargs: Additional parameters passed to DecisionTreeClassifier.
        """
        self.model: Optional[DecisionTreeClassifier] = None
        self.scaler: Optional[MinMaxScaler] = None
        self.imputer: Optional[SimpleImputer] = None
        self.input_type: Optional[str] = None
        # Stores the feature names *after* potential flattening (for numpy input)
        self.processed_feature_names: Optional[List[str]] = None 
        # Stores the original feature names before potential flattening
        self.original_feature_names_: Optional[List[str]] = None 
        self.sequence_length: Optional[int] = None
        self.n_original_features: Optional[int] = None
        self.label_col: Optional[str] = None
        
        # --- Validation related attributes ---
        self.validation_scores_: Dict[str, float] = {} # Stores validation metrics
        self.validation_set_size = kwargs.get('validation_set_size', 0.15) # Default 15%
        if not 0 <= self.validation_set_size < 1: # Allow 0 for no validation
            raise ValueError("validation_set_size must be between 0 (inclusive) and 1 (exclusive).")
        self.validation_metrics = kwargs.get('validation_metrics', ['accuracy', 'f1']) # Default metrics

        # --- Extract parameters from kwargs with defaults ---
        random_state = kwargs.get('random_state', 42)
        
        # Core Decision Tree parameters
        self.model_params = {
            'criterion': kwargs.get('criterion', 'gini'),
            'max_depth': kwargs.get('max_depth', None),
            'min_samples_split': kwargs.get('min_samples_split', 2),
            'min_samples_leaf': kwargs.get('min_samples_leaf', 1),
            'class_weight': kwargs.get('class_weight', 'balanced'),
            'random_state': random_state, # Use the stored random_state
            'max_features': kwargs.get('max_features', None)
        }
        if self.model_params.get('max_features') == 'None': # Check if the value is the string "None"
            print("INFO: Correcting 'max_features' parameter from string 'None' to Python None object.")
            self.model_params['max_features'] = None # Use direct assignment to update the value
        
        # Imputer strategy
        self._imputer_strategy = kwargs.get('imputer_strategy', 'mean')

        # Add any other kwargs intended for DecisionTreeClassifier, excluding validation ones
        allowed_dt_params = set(DecisionTreeClassifier().get_params().keys())
        excluded_params = {'validation_set_size', 'validation_metrics', 'imputer_strategy'}
        extra_dt_params = {
            k: v for k, v in kwargs.items() 
            if k in allowed_dt_params 
            and k not in self.model_params 
            and k not in excluded_params
        }
        self.model_params.update(extra_dt_params)

        if self.model_params.get('class_weight') is None:
            warnings.warn("class_weight was None, setting to 'balanced' by default.", UserWarning)
            self.model_params['class_weight'] = 'balanced'

        print(f"DecisionTreeModel Initialized with params: {self.model_params}")
        print(f"Imputer Strategy: {self._imputer_strategy}")
        print(f"Validation split: {self.validation_set_size*100}%, Metrics: {self.validation_metrics}")


    def _prepare_data_for_model(
        self, X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        label_col: Optional[str] = None, 
        is_training: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
        """
        Internal helper to preprocess data: reshape (if needed), scale, and impute NaNs.
        Sets internal attributes like input_type, sequence_length, n_original_features,
        processed_feature_names during training.
        Returns the final processed 2D NumPy array for the model, aligned labels (if training),
        and the list of processed feature names.
        """
        
        X_features_np = None # Holds 2D features before scaling/imputation
        y_aligned = None
        current_feature_names = None # Will hold the names corresponding to X_features_np columns
        
        # --- Stage 1: Determine input type, reshape (if needed), get initial features/labels ---
        if isinstance(X, pd.DataFrame):
            if is_training:
                if self.input_type is None: self.input_type = 'dataframe'
                elif self.input_type != 'dataframe': raise RuntimeError("Model already trained with different input type.")
                self.sequence_length = 1 

                if label_col is None or label_col not in X.columns:
                    raise ValueError(f"Label column '{label_col}' not found for DataFrame training.")
                self.label_col = label_col
                # Store original feature names
                original_names = X.columns.drop(label_col).tolist()
                if not original_names: raise ValueError("No feature columns found in DataFrame.")
                self.original_feature_names_ = original_names
                
                if self.n_original_features is None: self.n_original_features = len(original_names)
                elif self.n_original_features != len(original_names): raise ValueError("Feature count mismatch during training.")
                
                # For DataFrame input, processed names are the same as original names
                if self.processed_feature_names is None: self.processed_feature_names = original_names
                elif self.processed_feature_names != original_names: raise ValueError("Feature names mismatch during training.")

                current_feature_names = self.processed_feature_names
                X_features_np = X[current_feature_names].to_numpy()
                y_aligned = X[self.label_col].to_numpy()

                if X_features_np.shape[0] == 0: raise ValueError("No data rows found in DataFrame features.")

            else: # Detection/Scoring for DataFrame
                if self.input_type != 'dataframe' or self.scaler is None or self.imputer is None or self.original_feature_names_ is None or self.n_original_features is None:
                    raise RuntimeError("Model was not trained on DataFrame or required components (scaler/imputer/names) are missing.")
                
                missing_cols = set(self.original_feature_names_) - set(X.columns)
                if missing_cols: raise ValueError(f"Detection DataFrame missing required columns: {missing_cols}")

                current_feature_names = self.original_feature_names_ # Use original names for DF detection
                X_features_np = X[current_feature_names].to_numpy()

                if X_features_np.shape[0] == 0:
                    warnings.warn("No data rows provided for DataFrame detection/scoring.", RuntimeWarning)
                    # Return empty array matching expected 2D feature count
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

                # Assume original feature names are generic if not provided via run()
                if self.original_feature_names_ is None:
                    self.original_feature_names_ = [f"orig_feat_{i}" for i in range(self.n_original_features)]
                    warnings.warn("Original feature names not provided for NumPy input, generating generic names.", UserWarning)
                elif len(self.original_feature_names_) != self.n_original_features:
                     raise ValueError(f"Provided original feature names count ({len(self.original_feature_names_)}) != input features ({self.n_original_features}).")


                if y is None: raise ValueError("Labels 'y' are required for NumPy array training.")
                if not isinstance(y, np.ndarray) or y.ndim != 1 or len(y) != n_samples:
                    raise ValueError("Invalid 'y' array provided for NumPy training.")
                y_aligned = y

                n_flattened_features = seq_len * n_feat
                X_features_np = X.reshape(n_samples, n_flattened_features) # Flatten to 2D
                
                # Generate flattened feature names
                if self.processed_feature_names is None:
                    self.processed_feature_names = [f"{orig_name}_step_{j}" 
                                                    for j in range(seq_len) 
                                                    for orig_name in self.original_feature_names_]
                    if len(self.processed_feature_names) != n_flattened_features: # Fallback naming
                        self.processed_feature_names = [f"flat_feature_{k}" for k in range(n_flattened_features)]
                elif len(self.processed_feature_names) != n_flattened_features:
                    raise ValueError("Flattened feature name count mismatch during training.")
                current_feature_names = self.processed_feature_names

            else: # Detection/Scoring for NumPy
                if self.input_type != 'numpy' or self.scaler is None or self.imputer is None or self.processed_feature_names is None or self.original_feature_names_ is None or self.n_original_features is None or self.sequence_length is None:
                    raise RuntimeError("Model was not trained on NumPy or required components (scaler/imputer/names/dims) are missing.")

                if seq_len != self.sequence_length: raise ValueError(f"Input sequence length {seq_len} != train sequence length {self.sequence_length}.")
                if n_feat != self.n_original_features: raise ValueError(f"Input feature count {n_feat} != train feature count {self.n_original_features}.")

                current_feature_names = self.processed_feature_names # Use flattened names
                n_flattened_features = len(current_feature_names)
                if n_samples == 0:
                    warnings.warn("No samples provided for NumPy detection/scoring.", RuntimeWarning)
                    return np.empty((0, n_flattened_features)), None, current_feature_names

                X_features_np = X.reshape(n_samples, n_flattened_features) # Flatten to 2D
        else:
            raise TypeError("Input 'X' must be a pandas DataFrame or a 3D NumPy array.")

        # --- Stage 2: Scaling ---
        if is_training:
            self.scaler = MinMaxScaler()
            X_processed_scaled = self.scaler.fit_transform(X_features_np)
        else:
            if self.scaler is None or not hasattr(self.scaler, 'scale_'): 
                 raise RuntimeError("Scaler not fitted. Call run() first.")
            X_processed_scaled = self.scaler.transform(X_features_np)
        
        # --- Stage 3: Imputation ---
        if is_training:
            self.imputer = SimpleImputer(strategy=self._imputer_strategy)
            X_processed_imputed = self.imputer.fit_transform(X_processed_scaled)
        else:
            if self.imputer is None or not hasattr(self.imputer, 'statistics_'): 
                 raise RuntimeError("Imputer not fitted. Call run() first.")
            X_processed_imputed = self.imputer.transform(X_processed_scaled)
            
        # Check for remaining NaNs after imputation (should ideally be zero)
        if np.isnan(X_processed_imputed).any():
            warnings.warn("NaN values detected *after* imputation. Check input data or imputer strategy. Applying nan_to_num as fallback.", RuntimeWarning)
            # Apply nan_to_num as a final fallback
            X_processed_imputed = np.nan_to_num(X_processed_imputed, nan=0.0) 

        # Ensure current_feature_names is set
        if current_feature_names is None:
             raise RuntimeError("Internal Error: current_feature_names not set.")
             
        return X_processed_imputed, y_aligned, current_feature_names


    def run(self, X: Union[pd.DataFrame, np.ndarray], 
            y: Optional[np.ndarray] = None, 
            label_col: str = 'label',
            original_feature_names: Optional[List[str]] = None):
        """
        Prepares data, splits into train/validation, trains the Decision Tree 
        classifier on the training split, and evaluates on the validation split.

        Args:
            X: Input data (DataFrame or 3D NumPy). The *entire* training dataset.
            y: Target labels (required if X is NumPy array). Ignored if X is DataFrame.
            label_col: Name of the target label column (used if X is DataFrame).
            original_feature_names: List of original feature names before potential
                                     flattening (Required if X is a NumPy array).
        """
        print(f"Running training for DecisionTreeModel (Input type: {'DataFrame' if isinstance(X, pd.DataFrame) else 'NumPy'})...")
        
        # Store original feature names if provided (necessary for NumPy training)
        if isinstance(X, np.ndarray):
            if original_feature_names is None:
                raise ValueError("`original_feature_names` must be provided when training with NumPy input `X`.")
            if X.shape[2] != len(original_feature_names):
                 raise ValueError(f"NumPy input feature dimension ({X.shape[2]}) doesn't match length of original_feature_names ({len(original_feature_names)}).")
            self.original_feature_names_ = original_feature_names
            print(f"Stored {len(self.original_feature_names_)} original feature names for NumPy input.")
        elif original_feature_names is not None:
             warnings.warn("`original_feature_names` provided but input `X` is DataFrame. Names will be inferred from DataFrame columns.", UserWarning)
             # _prepare_data_for_model will extract names from DF columns

        # --- Step 1: Prepare data (sets internal attributes, scales, imputes) ---
        X_processed_imputed, y_aligned, _ = self._prepare_data_for_model(
            X, y=y, label_col=label_col, is_training=True
        )

        if y_aligned is None: # Should only happen if input was empty or malformed
            raise RuntimeError("No labels available for training after preprocessing.")
        if X_processed_imputed.shape[0] != len(y_aligned):
            raise RuntimeError(f"Mismatch between processed data samples ({X_processed_imputed.shape[0]}) and labels ({len(y_aligned)}).")

        # --- Step 2: Split data into Training and Validation Sets ---
        X_train, X_val, y_train, y_val = None, None, None, None
        
        # Only split if validation size > 0 and dataset is large enough
        if self.validation_set_size > 0 and X_processed_imputed.shape[0] > 1 / self.validation_set_size: 
            try:
                current_random_state = self.model_params.get('random_state') 
                
                X_train, X_val, y_train, y_val = train_test_split(
                    X_processed_imputed,
                    y_aligned,
                    test_size=self.validation_set_size,
                    random_state=current_random_state, 
                    stratify=y_aligned # Important for imbalanced datasets
                )
                print(f"Data split: Train shape={X_train.shape}, Validation shape={X_val.shape}")
                if X_val.shape[0] == 0: # Should be rare with check above, but possible
                     warnings.warn("Validation set is empty after split despite size check. Training on full data.", RuntimeWarning)
                     X_train, y_train = X_processed_imputed, y_aligned
                     X_val, y_val = None, None 
            except ValueError as e:
                warnings.warn(f"Could not stratify split (maybe only one class present or too few samples per class?): {e}. Training on full data, no validation scores.", RuntimeWarning)
                X_train, y_train = X_processed_imputed, y_aligned
                X_val, y_val = None, None
        else:
             if self.validation_set_size == 0:
                 print("Validation set size is 0. Training on full data, no validation scores.")
             else:
                  warnings.warn(f"Dataset too small ({X_processed_imputed.shape[0]} samples) for validation split size ({self.validation_set_size}). Training on full data.", RuntimeWarning)
             X_train, y_train = X_processed_imputed, y_aligned
             X_val, y_val = None, None 


        # Check labels in the *final* training split
        if y_train is None or len(y_train) == 0:
             raise RuntimeError("Training set is empty after split.")
        n_neg_train = np.sum(y_train == 0); n_pos_train = np.sum(y_train == 1)
        print(f"Training labels composition: {n_neg_train} negative, {n_pos_train} positive.")
        if n_pos_train == 0: warnings.warn(f"No positive samples ({self.label_col or 'label'}=1) found in the FINAL training split.", RuntimeWarning)
        if n_neg_train == 0: warnings.warn(f"No negative samples ({self.label_col or 'label'}=0) found in the FINAL training split.", RuntimeWarning)
        
        # --- Step 3: Training the Classifier on the Training Split ---
        print(f"Training DecisionTreeClassifier with {X_train.shape[0]} samples, {X_train.shape[1]} features...")
        print(f"Using model parameters: {self.model_params}")
        
        self.model = DecisionTreeClassifier(**self.model_params)

        try:
            # Fit ONLY on the training part
            self.model.fit(X_train, y_train)
        except Exception as e:
            raise RuntimeError(f"Decision Tree fitting failed: {e}") from e
            
        print("Model training complete.")
        if hasattr(self.model, 'classes_'): print(f"Model trained with classes: {self.model.classes_}")

        # --- Step 4: Evaluate on the Validation Split (if available) ---
        self.validation_scores_ = {} # Reset scores
        if X_val is not None and y_val is not None and X_val.shape[0] > 0:
            print(f"Evaluating model on validation set ({X_val.shape[0]} samples)...")
            try:
                y_pred_val = self.model.predict(X_val)
                y_proba_val = self.model.predict_proba(X_val)

                for metric_name in self.validation_metrics:
                    score = np.nan # Default score if calculation fails
                    try:
                        if metric_name == 'accuracy':
                            score = accuracy_score(y_val, y_pred_val)
                        elif metric_name == 'f1':
                            score = f1_score(y_val, y_pred_val, zero_division=0)
                        elif metric_name == 'roc_auc':
                            # Check if multiple classes exist in validation labels and predictions
                            if len(np.unique(y_val)) > 1 and len(np.unique(y_pred_val)) > 1 : 
                                # Ensure probabilities for positive class exist
                                pos_class_idx = np.where(self.model.classes_ == 1)[0]
                                if len(pos_class_idx) > 0 and y_proba_val.shape[1] > pos_class_idx[0]:
                                    score = roc_auc_score(y_val, y_proba_val[:, pos_class_idx[0]])
                                else: 
                                    warnings.warn(f"Positive class (1) not found in model classes ({self.model.classes_}) or probabilities shape ({y_proba_val.shape}) insufficient for ROC AUC calculation.", RuntimeWarning)
                            else:
                                warnings.warn(f"ROC AUC score is not defined for validation set (y_val unique: {np.unique(y_val)}, y_pred_val unique: {np.unique(y_pred_val)}). Requires at least two classes in both.", RuntimeWarning)
                                score = np.nan # AUC undefined for single class
                        else:
                            warnings.warn(f"Unsupported validation metric '{metric_name}'. Skipping.", UserWarning)
                            continue # Skip unsupported metric
                            
                        self.validation_scores_[metric_name] = score
                        print(f"  Validation {metric_name}: {score:.4f}")
                        
                    except Exception as metric_e:
                        print(f"  Failed to calculate validation metric '{metric_name}': {metric_e}")
                        self.validation_scores_[metric_name] = np.nan # Store NaN on error
                        
            except Exception as eval_e:
                print(f"Failed during validation set evaluation: {eval_e}")
        else:
            print("Skipping validation set evaluation (validation set not available or empty).")


    def get_anomaly_score(self, detection_data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Calculates anomaly scores (probability of class 1) for new data.
        """
        if self.model is None or self.scaler is None or self.imputer is None or self.input_type is None:
            raise RuntimeError("Model is not trained or key components are missing. Cannot score.")

        n_input_samples = len(detection_data) if isinstance(detection_data, pd.DataFrame) else detection_data.shape[0]
        final_scores = np.full(n_input_samples, np.nan, dtype=float)
        
        if n_input_samples == 0: 
            return final_scores 

        try:
            # Prepare data (handles input type, reshape, scaling, imputation)
            X_processed_imputed, _, processed_names = self._prepare_data_for_model(
                detection_data, is_training=False, label_col=self.label_col 
            )

            if X_processed_imputed.shape[0] == 0:
                warnings.warn("No processable data found for scoring after preparation.", RuntimeWarning)
                return final_scores 

            # Predict Probabilities using the fitted base model
            probabilities = self.model.predict_proba(X_processed_imputed)
            
            # Find the index corresponding to the positive class (1)
            if hasattr(self.model, 'classes_'):
                positive_class_index = np.where(self.model.classes_ == 1)[0]
                if len(positive_class_index) > 0:
                    class_index_to_use = positive_class_index[0]
                    # Check if index is valid for probabilities shape
                    if probabilities.shape[1] > class_index_to_use:
                         anomaly_scores = probabilities[:, class_index_to_use]
                    else:
                         warnings.warn(f"Positive class index {class_index_to_use} is out of bounds for probabilities shape {probabilities.shape}. Returning NaN.", RuntimeWarning)
                         anomaly_scores = np.full(X_processed_imputed.shape[0], np.nan) # Return NaNs if index invalid
                else: 
                    warnings.warn(f"Positive class (1) not found in model classes ({self.model.classes_}) during scoring. Returning probability of first class.", RuntimeWarning)
                    class_index_to_use = 0 
                    anomaly_scores = probabilities[:, class_index_to_use] if probabilities.shape[1] > 0 else np.zeros(X_processed_imputed.shape[0])
            else:
                warnings.warn("Model classes_ attribute not found. Assuming class 1 is the second column for probabilities.", RuntimeWarning)
                if probabilities.shape[1] < 2:
                    raise RuntimeError("Cannot determine anomaly score: predict_proba returned fewer than 2 columns and classes_ attribute is missing.")
                anomaly_scores = probabilities[:, 1] # Fallback assumption

            # Assign scores - length should match processed data length
            # Note: _prepare_data_for_model handles alignment for DataFrames
            # For NumPy, input/output length should match unless input was empty
            len_to_copy = min(len(anomaly_scores), n_input_samples)
            if len(anomaly_scores) != n_input_samples:
                 warnings.warn(f"Score length ({len(anomaly_scores)}) != input samples ({n_input_samples}). This might happen with sequence models or data issues. Aligning output.", RuntimeWarning)
            final_scores[:len_to_copy] = anomaly_scores[:len_to_copy]


        except Exception as e:
            warnings.warn(f"Anomaly score calculation failed: {e}. Returning NaNs for affected samples.", RuntimeWarning)
            traceback.print_exc() # Print details for debugging
            # final_scores is already initialized with NaNs
            
        return final_scores


    def detect(self, detection_data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Detects anomalies (predicts class 1) in new data.
        """
        if self.model is None or self.scaler is None or self.imputer is None or self.input_type is None:
            raise RuntimeError("Model is not trained or key components are missing. Cannot detect.")

        n_input_samples = len(detection_data) if isinstance(detection_data, pd.DataFrame) else detection_data.shape[0]
        final_anomalies = np.zeros(n_input_samples, dtype=bool) # Default to False

        if n_input_samples == 0: 
            return final_anomalies 

        try:
            # Prepare data (handles input type, reshape, scaling, imputation)
            X_processed_imputed, _, processed_names = self._prepare_data_for_model(
                detection_data, is_training=False, label_col=self.label_col
            )

            if X_processed_imputed.shape[0] == 0:
                warnings.warn("No processable data found for detection after preparation.", RuntimeWarning)
                return final_anomalies 

            # Prediction using the fitted base model
            predictions = self.model.predict(X_processed_imputed)
            anomalies = (predictions == 1)

            # Assign results - length should match processed data length
            # Note: _prepare_data_for_model handles alignment for DataFrames
            # For NumPy, input/output length should match unless input was empty
            len_to_copy = min(len(anomalies), n_input_samples)
            if len(anomalies) != n_input_samples:
                 warnings.warn(f"Detection length ({len(anomalies)}) != input samples ({n_input_samples}). This might happen with sequence models or data issues. Aligning output.", RuntimeWarning)
            final_anomalies[:len_to_copy] = anomalies[:len_to_copy]

        except Exception as e:
            warnings.warn(f"Anomaly detection failed: {e}. Returning False for affected samples.", RuntimeWarning)
            traceback.print_exc() # Print details for debugging
            # final_anomalies is already initialized with False

        return final_anomalies


    # --- METHOD FOR XAI (SHAP/LIME) ---
    def predict_proba(self, X_xai: np.ndarray) -> np.ndarray:
        """
        Prediction function for XAI methods (SHAP/LIME).
        Handles potential 3D input (if seq_len=1 for DF model) by reshaping.
        Ensures data is scaled and imputed before prediction.
        Returns probabilities for all classes.
        """
        if self.model is None or self.scaler is None or self.imputer is None or self.input_type is None \
            or self.n_original_features is None:
            raise RuntimeError("Model is not trained or key components (scaler/imputer/input_type/dims/names) are missing for XAI prediction.")
        
        if not isinstance(X_xai, np.ndarray):
            raise TypeError("Input X_xai for XAI must be a NumPy array.")

        n_instances = X_xai.shape[0]
        if n_instances == 0:
            n_classes = len(self.model.classes_) if hasattr(self.model, 'classes_') and self.model.classes_ is not None else 2
            return np.empty((0, n_classes))

        X_to_process_2d = None # Variable to hold the correctly shaped 2D data before scaling

        try:
            # --- Step 1: Validate input shape and reshape if necessary ---
            if self.input_type == 'dataframe':
                # DataFrame models expect 2D (n_instances, n_features) input internally,
                # but XAI wrappers might send 3D (n_instances, 1, n_features). Handle this.

                if X_xai.ndim == 3:
                    n_inst, seq_len, n_feat = X_xai.shape
                    # ALLOW 3D input IF seq_len is 1 for DataFrame models
                    if seq_len == 1:
                        if n_feat != self.n_original_features:
                             raise ValueError(f"XAI 3D input feature mismatch: got {n_feat} features, expected {self.n_original_features}")
                        # Reshape (n_instances, 1, n_features) to (n_instances, n_features)
                        print(f"DEBUG (predict_proba DF): Reshaping XAI input {X_xai.shape} to 2D.") 
                        X_to_process_2d = X_xai.reshape(n_inst, n_feat)
                    else:
                        # If seq_len is not 1, then it's an invalid 3D shape for DF model
                        raise ValueError(f"XAI 3D input for DataFrame model must have seq_len=1, got seq_len={seq_len}")

                elif X_xai.ndim == 2:
                    if X_xai.shape[1] != self.n_original_features:
                        raise ValueError(f"XAI 2D input has {X_xai.shape[1]} features, expected {self.n_original_features}")
                    X_to_process_2d = X_xai # Use 2D input directly
                else:
                    # Reject dimensions other than 2 or 3 (with seq_len=1)
                    raise ValueError(f"XAI input must be 2D or 3D (with seq_len=1) for DataFrame-trained model, got {X_xai.ndim}D.")

            elif self.input_type == 'numpy':
                # NumPy models expect 3D (n_instances, seq_len, features) input from XAI and flatten it.
                if self.sequence_length is None: raise RuntimeError("Sequence length not set for NumPy-trained model.")
                if X_xai.ndim != 3:
                    raise ValueError(f"XAI input must be 3D (n_instances, seq_len, features) for NumPy-trained model, got {X_xai.ndim}D.")

                n_inst_np, seq_len_np, n_feat_np = X_xai.shape # Use different var names
                if seq_len_np != self.sequence_length: raise ValueError(f"XAI input seq len ({seq_len_np}) != train seq len ({self.sequence_length}).")
                if n_feat_np != self.n_original_features: raise ValueError(f"XAI input features ({n_feat_np}) != train features ({self.n_original_features}).")

                # Flatten 3D input to 2D for internal model
                n_flattened = seq_len_np * n_feat_np
                # print(f"DEBUG (predict_proba NP): Reshaping XAI input {X_xai.shape} to 2D ({n_inst_np}, {n_flattened}).") # Optional debug
                X_to_process_2d = X_xai.reshape(n_inst_np, n_flattened)

            else:
                raise RuntimeError(f"Unsupported training input_type '{self.input_type}' for XAI.")

            # --- Step 2: Scaling ---
            if X_to_process_2d is None: # Should not happen if logic above is correct
                raise RuntimeError("Internal error: X_to_process_2d is None before scaling.")
            if self.scaler is None or not hasattr(self.scaler, 'scale_'):
                raise RuntimeError("Scaler has not been fitted.")
            X_scaled = self.scaler.transform(X_to_process_2d)

            # --- Step 3: Imputation ---
            if self.imputer is None or not hasattr(self.imputer, 'statistics_'):
                raise RuntimeError("Imputer has not been fitted.")
            X_imputed = self.imputer.transform(X_scaled)

            # Final check for NaNs before prediction in XAI context
            if np.isnan(X_imputed).any():
                warnings.warn("NaNs detected in XAI input *after* imputation. Applying nan_to_num.", RuntimeWarning)
                X_imputed = np.nan_to_num(X_imputed, nan=0.0) # Fallback for XAI

            # --- Step 4: Predict probabilities using the internal model ---
            if X_imputed is None: # Should not happen
                 raise RuntimeError("Internal error: Imputed data for XAI prediction is None.")

            probabilities = self.model.predict_proba(X_imputed)

        except Exception as e:
            # Catch errors during preprocessing or prediction within XAI
            # Log detailed error before raising generic one
            print(f"ERROR during predict_proba execution: {type(e).__name__} - {e}")
            traceback.print_exc() # Print full traceback for debugging
            raise RuntimeError(f"XAI prediction failed during preprocessing or model prediction. Error: {e}") from e

        # --- Step 5: Validate output shape ---
        # Check if model is fitted and has classes_ attribute
        if not hasattr(self.model, 'classes_') or self.model.classes_ is None:
             # This can happen if fit failed or model doesn't expose classes_
             warnings.warn("Model classes_ attribute not available. Cannot validate output probability shape accurately.", RuntimeWarning)
             expected_cols = None 
        else:
             expected_cols = len(self.model.classes_)
        
        # Perform shape validation if possible
        if expected_cols is not None:
            if probabilities.ndim != 2 or probabilities.shape[0] != n_instances or probabilities.shape[1] != expected_cols:
                warnings.warn(f"predict_proba output shape {probabilities.shape} unexpected. Expected ({n_instances}, {expected_cols}). Check model.", RuntimeWarning)
                # Decide how to handle this? Return as is? Raise error? Pad?
                # For now, returning the potentially incorrect shape with a warning.

        return probabilities


    # --- Optional: Method to get validation scores ---
    def get_validation_scores(self) -> Dict[str, float]:
        """Returns the computed validation scores."""
        if not hasattr(self, 'validation_scores_') or not self.validation_scores_:
             print("Validation scores not available (model not run, validation failed, or validation_set_size=0).")
             return {}
        return self.validation_scores_
        
    # --- Optional: Method to get feature importances ---
    def get_feature_importances(self) -> Optional[Dict[str, float]]:
        """Returns feature importances from the trained decision tree, 
           using the processed (potentially flattened) feature names."""
        if self.model is None:
            print("Model not trained yet.")
            return None
            
        if not hasattr(self.model, 'feature_importances_'):
            print("Model does not have feature_importances_ attribute.")
            return None

        # Use the processed_feature_names which correspond to the model's input
        if self.processed_feature_names is None:
            print("Processed feature names not available.")
            # Return importances without names if names aren't stored
            return {f"feature_{i}": imp for i, imp in enumerate(self.model.feature_importances_)}

        importances = self.model.feature_importances_
        if len(importances) != len(self.processed_feature_names):
            warnings.warn(f"Mismatch between number of importances ({len(importances)}) and processed feature names ({len(self.processed_feature_names)}). Returning potentially misaligned results.")
            # Try to return anyway, might be misaligned
            max_len = min(len(importances), len(self.processed_feature_names))
            return {self.processed_feature_names[i]: importances[i] for i in range(max_len)}
            
        return dict(zip(self.processed_feature_names, importances))