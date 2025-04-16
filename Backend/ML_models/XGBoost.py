import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from ML_models import model_interface 
from typing import List, Dict, Optional
import warnings

class XGBoostModel(model_interface.ModelInterface):
    """
    Supervised XGBoost classification model for anomaly detection using labeled data.

    Uses lagged features based on `TIME_STEPS` for temporal context.
    Trains an XGBClassifier to predict the anomaly label.
    Handles class imbalance using 'scale_pos_weight'.
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, **kwargs):
        """
        Initializes the XGBoost classifier model.

        Args:
            n_estimators (int): Number of boosting rounds (trees).
            learning_rate (float): Step size shrinkage.
            max_depth (int): Maximum depth of a tree.
            random_state (int): Random seed for reproducibility.
            **kwargs: Additional parameters passed directly to XGBClassifier.
        """
        self.model: Optional[xgb.XGBClassifier] = None
        self.scaler: Optional[MinMaxScaler] = None
        self.original_feature_names: Optional[List[str]] = None # Features only (no label)
        self.all_feature_names: Optional[List[str]] = None    # Original + Lagged features
        self.time_steps: Optional[int] = None
        self.label_col: Optional[str] = None # Name of the target label column
        self.sequence_length: Optional[int] = None

        # Store base model parameters, allow overrides via kwargs
        self.model_params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'objective': 'binary:logistic', # For binary classification
            'eval_metric': 'logloss',       # Common metric for binary classification
            'use_label_encoder': False,     # Recommended practice with modern XGBoost
            'random_state': random_state,
            'n_jobs': -1 # Use all available CPU cores
        }
        self.model_params.update(kwargs) # Apply any additional user-provided parameters

        print(f"XGBoostSupervisedModel Initialized with base params: {self.model_params}")
        print("Note: 'scale_pos_weight' will be calculated during run() based on data.")


    def _create_lagged_features(self, df_features: pd.DataFrame, time_steps: int) -> pd.DataFrame:
        """ Creates lagged features from the feature DataFrame and handles resulting NaNs. """
        if time_steps <= 0:
            return df_features.copy() # No lagging needed

        df_lagged = df_features.copy()
        original_cols = df_features.columns.tolist()
        print(f"Creating lagged features for {time_steps} steps...")

        for t in range(1, time_steps + 1):
            shifted = df_features[original_cols].shift(t)
            shifted.columns = [f"{col}_lag_{t}" for col in original_cols]
            df_lagged = pd.concat([df_lagged, shifted], axis=1)

        # Drop rows with NaNs introduced by lagging
        original_len = len(df_lagged)
        df_lagged.dropna(inplace=True)
        new_len = len(df_lagged)

        if original_len > 0:
            print(f"Dropped {original_len - new_len} rows due to NaN values after lagging.")
        
        if new_len == 0 and original_len > 0:
             raise ValueError(f"Data is too short ({original_len} rows) to create lagged features for {time_steps} steps. No data remains after dropping NaNs.")
        elif new_len == 0:
             warnings.warn("Input DataFrame to _create_lagged_features was empty.", RuntimeWarning)
             
        return df_lagged

    def run(self, df: pd.DataFrame, time_steps: int = 5, label_col: str = 'label'):
        """
        Preprocesses data, creates lagged features, trains the XGBoost classifier
        on features and labels.

        Args:
            df (pd.DataFrame): Input DataFrame containing features AND the label column.
            time_steps (int): Number of past time steps for lagged features. Defaults to 5.
            label_col (str): Name of the column containing the anomaly labels
                             (e.g., 0 for normal, 1 for anomaly). Defaults to 'is_anomaly'.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input 'df' must be a pandas DataFrame.")
        if df.empty:
             raise ValueError("Input DataFrame 'df' is empty.")
        if time_steps < 0:
             raise ValueError("time_steps cannot be negative.")
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in DataFrame columns: {df.columns.tolist()}")

        self.time_steps = time_steps
        self.label_col = label_col
        self.sequence_length = self.time_steps
        self.original_feature_names = df.columns.drop(label_col).tolist()
        if not self.original_feature_names:
            raise ValueError("DataFrame must have at least one feature column besides the label column.")

        print(f"Running XGBoostSupervisedModel training with time_steps={self.time_steps}, label='{self.label_col}'...")

        # 1. Separate Features and Labels
        X_orig = df[self.original_feature_names]
        y = df[self.label_col]

        # 2. Feature Engineering: Create lagged features for X
        X_processed = self._create_lagged_features(X_orig, self.time_steps)

        if X_processed.empty:
             warnings.warn("No data available for training after creating lagged features and dropping NaNs.", RuntimeWarning)
             print("Warning: Training aborted.")
             # Reset state to indicate not trained
             self.model = None
             self.scaler = None
             self.all_feature_names = None
             return # Cannot train

        self.all_feature_names = X_processed.columns.tolist() # Store all feature names used

        # 3. Align Labels with Processed Features
        # Use the index of X_processed to select the corresponding labels from y
        y_aligned = y.loc[X_processed.index]
        print(f"Aligned data: {len(X_processed)} samples for training.")


        # 4. Scaling Features
        print("Scaling features...")
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X_processed)
        # Note: XGBoost can handle unscaled data, but scaling is generally good practice
        # and helps consistency if comparing with other models like LSTM.

        # 5. Handle Class Imbalance (Crucial for Anomaly Detection)
        n_neg = (y_aligned == 0).sum()
        n_pos = (y_aligned == 1).sum()
        
        if n_pos == 0:
            warnings.warn(f"No positive samples ({label_col}=1) found in the aligned training data. Model may not learn to detect anomalies effectively. Ensure your training data includes anomalies.", RuntimeWarning)
            # Set scale_pos_weight to 1 if no positive samples, or consider raising an error
            scale_pos_weight = 1 
        elif n_neg == 0:
             warnings.warn(f"No negative samples ({label_col}=0) found in the aligned training data. Model may not generalize well.", RuntimeWarning)
             scale_pos_weight = 1 # Or handle as appropriate
        else:
            scale_pos_weight = n_neg / n_pos
            print(f"Calculated scale_pos_weight for imbalance: {scale_pos_weight:.2f} ({n_neg} neg / {n_pos} pos)")

        # Add scale_pos_weight to model parameters
        current_model_params = self.model_params.copy()
        current_model_params['scale_pos_weight'] = scale_pos_weight

        # 6. Training the Classifier
        print(f"Training XGBClassifier with parameters: {current_model_params}...")
        self.model = xgb.XGBClassifier(**current_model_params)

        self.model.fit(X_scaled, y_aligned)
        print("Model training complete.")
        
    def get_anomaly_score(self, detection_df: pd.DataFrame) -> np.ndarray:
        """
        Calculates anomaly scores for new data using the trained classifier's
        probability estimates for the anomaly class (class 1).

        Args:
            detection_df (pd.DataFrame): DataFrame with new data points to check.
                                         Must have the same original feature columns
                                         as the training data (excluding the label column).
                                         Needs sufficient history for lags if time_steps > 0.

        Returns:
            np.ndarray: A 1D float array with the same length as detection_df.
                        Contains the predicted probability of the anomaly class (1).
                        Higher values indicate higher likelihood of anomaly.
                        Rows where score couldn't be computed (due to lags)
                        will have NaN.
        """
        if self.model is None or self.scaler is None or self.original_feature_names is None or self.all_feature_names is None or self.time_steps is None:
            raise RuntimeError("Model is not trained or ready. Call run() first.")
        if not isinstance(detection_df, pd.DataFrame):
            raise TypeError("Input 'detection_df' must be a pandas DataFrame.")

        if detection_df.empty:
            return np.array([], dtype=float)

        # Check if required original feature columns exist
        missing_cols = set(self.original_feature_names) - set(detection_df.columns)
        if missing_cols:
            raise ValueError(f"Detection DataFrame is missing required feature columns: {missing_cols}")
            
        # Select only the original feature columns for processing
        X_detect_orig = detection_df[self.original_feature_names]
        original_indices = detection_df.index # Keep track before lagging

        print(f"Calculating anomaly scores for {len(detection_df)} samples...")

        # 1. Feature Engineering (Consistent with run/detect)
        X_detect_processed = self._create_lagged_features(X_detect_orig, self.time_steps)

        if X_detect_processed.empty:
             warnings.warn("No data remained for scoring after creating lagged features and dropping NaNs.", RuntimeWarning)
             # Return NaNs for the original input length
             return np.full(len(detection_df), np.nan, dtype=float)

        processed_indices = X_detect_processed.index # Indices that survived lagging

        # 2. Scaling (Use fitted scaler)
        try:
            # Ensure columns match exactly those used in training (original + lagged)
            X_detect_scaled = self.scaler.transform(X_detect_processed[self.all_feature_names])
        except ValueError as e:
             raise ValueError(f"Failed to scale detection data. Check columns. Expected: {self.all_feature_names}. Error: {e}") from e
        except Exception as e:
             raise RuntimeError(f"An unexpected error occurred during scaling detection data: {e}") from e

        # 3. Prediction of Probabilities
        print(f"Predicting anomaly probabilities for {len(X_detect_scaled)} processed samples...")
        try:
            # Use predict_proba to get [P(class_0), P(class_1)] for each sample
            probabilities = self.model.predict_proba(X_detect_scaled) 
        except Exception as e:
            raise RuntimeError(f"Model probability prediction failed. Input shape: {X_detect_scaled.shape}. Error: {e}") from e

        # Extract the probability of the anomaly class (class 1, which is the second column)
        # probabilities array shape is (n_samples, n_classes), we want column index 1
        anomaly_scores_processed = probabilities[:, 1]

        # 4. Align scores back to the original DataFrame index
        results_series = pd.Series(anomaly_scores_processed, index=processed_indices)
        # Reindex to original DataFrame length, fill non-computable rows with NaN
        final_scores = results_series.reindex(original_indices, fill_value=np.nan).values

        num_processed = len(processed_indices) 
        num_nan = np.isnan(final_scores).sum() # Count NaNs

        print(f"Score calculation complete. Generated scores for {num_processed} processable samples.")
        if num_nan > 0:
             print(f"  ({num_nan} samples could not be scored due to insufficient history for lags, assigned NaN score).")
             
        if len(final_scores) != len(detection_df):
             # Fallback, though reindex should handle this
             warnings.warn(f"Output score length mismatch ({len(final_scores)}) vs input ({len(detection_df)}).", RuntimeWarning)
             # Return scores matching processed length, or handle differently
             return anomaly_scores_processed 

        return final_scores # Return float array matching original detection_df length (with NaNs possible)


    def detect(self, detection_df: pd.DataFrame) -> np.ndarray:
        """
        Detects anomalies in new (unlabeled) data using the trained classifier.

        Args:
            detection_df (pd.DataFrame): DataFrame with new data points to check.
                                         Must have the same original feature columns
                                         as the training data (excluding the label column).
                                         Needs sufficient history for lags if time_steps > 0.

        Returns:
            np.ndarray: A 1D boolean array with the same length as detection_df.
                        True indicates a predicted anomaly (class 1), False for normal (class 0).
                        Rows where prediction couldn't be made (due to lags)
                        will be False.
        """
        if self.model is None or self.scaler is None or self.original_feature_names is None or self.all_feature_names is None or self.time_steps is None:
            raise RuntimeError("Model is not trained or ready. Call run() first.")
        if not isinstance(detection_df, pd.DataFrame):
            raise TypeError("Input 'detection_df' must be a pandas DataFrame.")

        if detection_df.empty:
            return np.array([], dtype=bool)

        # Check if required original feature columns exist
        missing_cols = set(self.original_feature_names) - set(detection_df.columns)
        if missing_cols:
            raise ValueError(f"Detection DataFrame is missing required feature columns: {missing_cols}")
            
        # Select only the original feature columns for processing
        X_detect_orig = detection_df[self.original_feature_names]
        original_indices = detection_df.index # Keep track before lagging

        print(f"Detecting anomalies in {len(detection_df)} samples...")

        # 1. Feature Engineering (Consistent with run)
        X_detect_processed = self._create_lagged_features(X_detect_orig, self.time_steps)

        if X_detect_processed.empty:
             warnings.warn("No data remained for detection after creating lagged features and dropping NaNs.", RuntimeWarning)
             # Return all False for the original input length
             return np.zeros(len(detection_df), dtype=bool)

        processed_indices = X_detect_processed.index # Indices that survived lagging

        # 2. Scaling (Use fitted scaler)
        try:
            # Ensure columns match exactly those used in training (original + lagged)
            X_detect_scaled = self.scaler.transform(X_detect_processed[self.all_feature_names])
        except ValueError as e:
             raise ValueError(f"Failed to scale detection data. Check columns. Expected: {self.all_feature_names}. Error: {e}") from e
        except Exception as e:
             raise RuntimeError(f"An unexpected error occurred during scaling detection data: {e}") from e

        # 3. Prediction
        print(f"Predicting labels for {len(X_detect_scaled)} processed samples...")
        try:
            predictions = self.model.predict(X_detect_scaled) # Predicts class labels (0 or 1)
        except Exception as e:
            raise RuntimeError(f"Model prediction failed. Input shape: {X_detect_scaled.shape}. Error: {e}") from e


        # Convert predictions (0/1) to boolean (False/True)
        anomalies_processed = (predictions == 1)

        # 4. Align predictions back to the original DataFrame index
        results_series = pd.Series(anomalies_processed, index=processed_indices)
        # Reindex to original DataFrame length, fill non-computable rows with False (not anomaly)
        final_anomalies = results_series.reindex(original_indices, fill_value=False).values

        num_anomalies = final_anomalies.sum()
        num_processed = len(processed_indices) # How many were actually scored
        num_nan_equivalent = len(detection_df) - num_processed

        print(f"Detection complete. Found {num_anomalies} anomalies out of {len(detection_df)} total samples.")
        if num_nan_equivalent > 0:
             print(f"  ({num_nan_equivalent} samples could not be processed due to insufficient history for lags, marked as non-anomaly).")
             
        if len(final_anomalies) != len(detection_df):
             # This case should be less likely with reindexing, but as a fallback:
             warnings.warn(f"Output length mismatch ({len(final_anomalies)}) vs input ({len(detection_df)}). Returning potentially misaligned results.", RuntimeWarning)
             # You might return a padded array or raise error depending on strictness needed
             # Returning the computed array here, but caller beware.
             return anomalies_processed # Return array matching processed rows length

        return final_anomalies # Return boolean array matching original detection_df length