# File: svm_autoencoder_model.py (Corrected Version)

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Assuming model_interface defines ModelInterface base class
from ML_models import model_interface
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, MinMaxScaler # Add MinMaxScaler if used by AE
from tensorflow.keras.models import Model # Use tensorflow.keras
from tensorflow.keras.layers import Input, Dense # Use tensorflow.keras
from typing import Union, List, Any
import warnings

class SVMModel(model_interface.ModelInterface):
    """
    Anomaly detection using an Autoencoder + OneClassSVM.
    Operates on 2D data (samples, features). Refactored for correctness & XAI.
    """
    # --- Make sequence_length an attribute for consistency, though unused internally ---
    # This value isn't used by run/detect logic but might be useful elsewhere.
    # It will be DIFFERENT from the sequence_length used by the XAI framework.
    sequence_length = 1 # Indicates it processes samples individually

    def __init__(self, encoding_dim: int = 10, svm_nu: float = 0.1, svm_kernel: str = 'rbf', svm_gamma: str = 'scale'):
        self.svm_model = OneClassSVM(kernel=svm_kernel, gamma=svm_gamma, nu=svm_nu)
        self.scaler = StandardScaler() # Or MinMaxScaler(feature_range=(0, 1)) if AE uses sigmoid
        self.encoder: Optional[Model] = None
        self.autoencoder: Optional[Model] = None
        self.threshold: Optional[float] = None
        self.n_features: Optional[int] = None
        self.encoding_dim = encoding_dim
        print(f"SVMModel Initialized (Encoding Dim: {encoding_dim}, SVM nu: {svm_nu}).")

    def __build_autoencoder(self, input_dim):
        print(f"Building Autoencoder: input_dim={input_dim}, encoding_dim={self.encoding_dim}")
        input_layer = Input(shape=(input_dim,), name='input_layer')
        encoded = Dense(self.encoding_dim, activation='relu', name='encoder_output')(input_layer)
        # Use linear output activation if using StandardScaler, sigmoid if MinMaxScaler(0,1)
        decoded = Dense(input_dim, activation='linear', name='decoder_output')(encoded)
        autoencoder = Model(inputs=input_layer, outputs=decoded, name='autoencoder')
        encoder = Model(inputs=input_layer, outputs=encoded, name='encoder')
        autoencoder.compile(optimizer='adam', loss='mse')
        print("Autoencoder Architecture:")
        autoencoder.summary(print_fn=print) # Ensure summary prints
        return autoencoder, encoder

    def run(self, df: pd.DataFrame, epochs: int = 10, batch_size: int = 32):
        if not isinstance(df, pd.DataFrame): raise TypeError("Input 'df' must be a pandas DataFrame.")
        if df.empty: raise ValueError("Input DataFrame for training is empty.")
        print(f"Running SVMModel training on data shape: {df.shape}")
        self.n_features = df.shape[1]
        if self.n_features == 0: raise ValueError("Input DataFrame has no feature columns.")

        # --- Scaling ---
        print("Fitting and transforming scaler...")
        X_train_scaled = self.scaler.fit_transform(df)
        X_train_scaled = X_train_scaled.astype(np.float32)

        # --- Train Autoencoder ---
        self.autoencoder, self.encoder = self.__build_autoencoder(self.n_features)
        print(f"Fitting Autoencoder for {epochs} epochs...")
        # Consider adding callbacks like EarlyStopping if needed
        self.autoencoder.fit(X_train_scaled, X_train_scaled,
                             epochs=epochs, batch_size=batch_size,
                             validation_split=0.1, shuffle=True, verbose=1)
        print("Autoencoder fitting complete.")

        # --- Get Encoded Representation & Train SVM ---
        print("Encoding training data...")
        train_encoded_data = self.encoder.predict(X_train_scaled)
        print(f"Encoded training data shape: {train_encoded_data.shape}")
        print("Fitting OneClassSVM...")
        self.svm_model.fit(train_encoded_data)
        print("OneClassSVM fitting complete.")

        # --- Set Threshold ---
        print("Calculating anomaly threshold...")
        decision_values_train = self.svm_model.decision_function(train_encoded_data)
        self.threshold = np.percentile(decision_values_train, 100 * self.svm_model.nu)
        print(f"Anomaly threshold set to: {self.threshold:.6f}")
        print("--- SVMModel Training Finished ---")


    def _preprocess_and_encode(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """ Internal helper: Scales (using fitted scaler) and encodes input data. """
        if self.scaler is None or self.encoder is None:
            raise RuntimeError("Model is not trained (scaler/encoder missing). Call run() first.")

        if isinstance(data, pd.DataFrame):
            if data.shape[1] != self.n_features: raise ValueError(f"Input data has {data.shape[1]} feats, expected {self.n_features}.")
            input_np = data.values
        elif isinstance(data, np.ndarray):
            if data.ndim == 1: data = data.reshape(1, -1)
            if data.ndim != 2: raise ValueError(f"NumPy input must be 2D (samples, features), got {data.ndim}D.")
            if data.shape[1] != self.n_features: raise ValueError(f"NumPy input has {data.shape[1]} feats, expected {self.n_features}.")
            input_np = data
        else: raise TypeError("Input must be a pandas DataFrame or a 2D NumPy array.")

        #print(f"Preprocessing {input_np.shape[0]} samples...")
        data_scaled = self.scaler.transform(input_np) # Use TRANSFORM only
        data_scaled = data_scaled.astype(np.float32)

        #print(f"Encoding {data_scaled.shape[0]} scaled samples...")
        encoded_data = self.encoder.predict(data_scaled)
        #print(f"Encoded data shape: {encoded_data.shape}")
        return encoded_data


    def get_anomaly_score(self, detection_data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """ Calculates SVM decision function scores (lower = more anomalous). Expects 2D input. """
        #print("Calculating anomaly scores...")
        encoded_data = self._preprocess_and_encode(detection_data)
        scores = self.svm_model.decision_function(encoded_data)
        #print(f"Calculated {len(scores)} scores.")
        return scores # Returns 1D array


    def detect(self, detection_data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """ Detects anomalies based on threshold. Expects 2D input. """
        if self.threshold is None: raise RuntimeError("Threshold not set. Call run() first.")
        scores = self.get_anomaly_score(detection_data)
        anomalies = scores < self.threshold
        print(f"Detected {np.sum(anomalies)} anomalies using threshold {self.threshold:.6f}.")
        return anomalies # Returns 1D boolean array
    
    def get_anomaly_score(self, detection_data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Calculates the anomaly score (SVM decision function) for input data.
        Operates on the encoded representation. Lower scores indicate higher anomaly likelihood.
        Expects 2D input (samples, features).

        Args:
            detection_data (Union[pd.DataFrame, np.ndarray]): Input data (samples, features).

        Returns:
            np.ndarray: The decision function scores (1D array, shape: (n_samples,)).
        """
        # Reuse the internal preprocessing and encoding helper
        # This ensures scaling and encoding are done consistently
        print("Calculating anomaly scores via get_anomaly_score...")
        # Ensure model is trained before calling helper
        if self.scaler is None or self.encoder is None or self.svm_model is None:
             raise RuntimeError("Model components (scaler/encoder/svm) not available. Call run() first.")

        encoded_data = self._preprocess_and_encode(detection_data) # Gets 2D encoded data

        # Get SVM score on encoded data - lower means more anomalous
        # Ensure svm_model is fitted (checked implicitly by _preprocess_and_encode checking encoder)
        if not hasattr(self.svm_model, "decision_function"):
             raise RuntimeError("Internal SVM model is not fitted or invalid.")

        scores = self.svm_model.decision_function(encoded_data)
        # print(f"Calculated {len(scores)} scores.") # Optional print
        return scores # Return the raw scores (1D array)

    # Ensure the _preprocess_and_encode helper method also exists in your class
    def _preprocess_and_encode(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """ Internal helper: Scales (using fitted scaler) and encodes input data. """
        if self.scaler is None or self.encoder is None:
            raise RuntimeError("Model is not trained (scaler or encoder missing). Call run() first.")
        # ...(rest of the implementation as provided before)...
        # Scale -> Ensure float32 -> Encode -> Return encoded_data
        # ... (previous implementation) ...
        if isinstance(data, pd.DataFrame):
            if data.shape[1] != self.n_features: raise ValueError(f"Input data has {data.shape[1]} feats, expected {self.n_features}.")
            input_np = data.values
        elif isinstance(data, np.ndarray):
            if data.ndim == 1: data = data.reshape(1, -1)
            if data.ndim != 2: raise ValueError(f"NumPy input must be 2D (samples, features), got {data.ndim}D.")
            if data.shape[1] != self.n_features: raise ValueError(f"NumPy input has {data.shape[1]} feats, expected {self.n_features}.")
            input_np = data
        else: raise TypeError("Input must be a pandas DataFrame or a 2D NumPy array.")
        data_scaled = self.scaler.transform(input_np)
        data_scaled = data_scaled.astype(np.float32)
        encoded_data = self.encoder.predict(data_scaled)
        return encoded_data