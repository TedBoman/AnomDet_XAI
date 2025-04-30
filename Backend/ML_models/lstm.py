import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
# Assuming model_interface exists and defines ModelInterface base class
from ML_models import model_interface
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from typing import Optional, Union, List # Make sure to import Union
import warnings # Import warnings

class LSTMModel(model_interface.ModelInterface):

    # Initializes the model and internal state
    def __init__(self):
        self.model = None
        self.scaler = None
        self.threshold = None
        # Initialize sequence_length, will be set during run()
        self.sequence_length = None
        print("LSTMModel Initialized (sequence_length will be set during run).")

    # Preprocesses, trains and fits the model
    def run(self, df, time_steps=10, epochs=2): # Changed default time_steps
        """
        Preprocesses data, builds, trains, and fits the LSTM autoencoder model.

        Args:
            df (pd.DataFrame): Input DataFrame containing features for training.
                               Assumes columns are features.
            time_steps (int): The sequence length (lookback window) to use for
                              creating training sequences. Defaults to 10.
        """
        if not isinstance(df, pd.DataFrame):
             raise TypeError("Input 'df' must be a pandas DataFrame.")
        if time_steps <= 0:
             raise ValueError("time_steps must be positive.")

        print(f"Running LSTMModel training with time_steps={time_steps}...")
        # --- Store the sequence length ---
        self.sequence_length = time_steps
        # ---

        features = df.shape[1]
        if features == 0:
             raise ValueError("Input DataFrame has no columns (features).")

        # --- Define Keras Model (Corrected Input Shape) ---
        # Input shape should be (sequence_length, features)
        inputs = Input(shape=(self.sequence_length, features))
        # Encoder LSTM - processes the sequence
        encoded = LSTM(64, activation='relu', return_sequences=False)(inputs)

        # Repeat the encoded vector for each time step for the decoder input
        decoded = RepeatVector(self.sequence_length)(encoded)
        # Decoder LSTM - reconstructs the sequence
        decoded = LSTM(64, activation='relu', return_sequences=True)(decoded)
        # Output layer for each time step
        outputs = TimeDistributed(Dense(features))(decoded) # Output features should match input features

        autoencoder = Model(inputs, outputs)
        autoencoder.compile(optimizer='adam', loss='mse')
        self.model = autoencoder
        print("LSTM Autoencoder Model Compiled:")
        self.model.summary() # Print model summary
        # --- End Model Definition ---

        # --- Data Preprocessing ---
        self.scaler = MinMaxScaler()
        data_normalized = self.scaler.fit_transform(df)

        # Create sequences using the stored sequence_length
        print(f"Creating sequences with length {self.sequence_length}...")
        X = self.__create_sequences(data_normalized, self.sequence_length)
        if X.size == 0:
             raise ValueError(f"Data is too short ({len(df)} rows) to create sequences of length {self.sequence_length}.")
        print(f"Created {X.shape[0]} sequences with shape {X.shape[1:]}")
        # --- End Data Preprocessing ---

        # --- Training ---
        # Simple split for threshold calculation (adjust if needed)
        train_size = int(len(X) * 0.8)
        if train_size == 0 and len(X) > 0: train_size = 1 # Ensure at least one sample if possible
        X_train = X[:train_size]
        X_test_threshold = X[train_size:] # Use test split for threshold

        if X_train.size == 0:
             warnings.warn("Training split is empty, model cannot be trained.", RuntimeWarning)
             # Handle this case: maybe raise error or set dummy threshold?
             self.threshold = np.inf # Set a default threshold if no training occurs
             return # Cannot proceed with training

        print(f"Fitting model on {X_train.shape[0]} training sequences...")
        self.model.fit(
            X_train, X_train, # Autoencoder learns to reconstruct input
            epochs=epochs, # Consider making epochs configurable
            batch_size=256, # Consider making batch_size configurable
            validation_split=0.2, # Use part of X_train for validation during training
            verbose=1,
            shuffle=True # Shuffle training data each epoch
        )
        print("Model fitting complete.")
        # --- End Training ---

        # --- Threshold Calculation ---
        if X_test_threshold.size > 0:
             print(f"Calculating threshold on {X_test_threshold.shape[0]} test sequences...")
             reconstructed = self.model.predict(X_test_threshold)
             reconstruction_error = np.mean(np.square(X_test_threshold - reconstructed), axis=(1, 2))
             # Use a percentile of the reconstruction error on test data as threshold
             self.threshold = np.percentile(reconstruction_error, 95) # Example: 95th percentile
             print(f"Anomaly threshold set to: {self.threshold:.6f}")
        else:
             warnings.warn("Test split for threshold calculation is empty. Threshold may be unreliable.", RuntimeWarning)
             # Fallback: use threshold from training reconstruction error (less ideal)
             if X_train.size > 0:
                 reconstructed_train = self.model.predict(X_train)
                 reconstruction_error_train = np.mean(np.square(X_train - reconstructed_train), axis=(1, 2))
                 self.threshold = np.percentile(reconstruction_error_train, 95)
                 print(f"Anomaly threshold set from training data: {self.threshold:.6f}")
             else: # Should have been caught earlier
                  self.threshold = np.inf
                  print("Error: Cannot set threshold - no data available.")

    def __create_sequences(self, data: np.ndarray, sequence_length: int) -> np.ndarray:
        """ Helper method to create 3D windowed sequences from 2D data. """
        sequences = []
        n_samples_total = data.shape[0]
        if n_samples_total < sequence_length:
            print(f"Warning in __create_sequences: Data length ({n_samples_total}) < sequence_length ({sequence_length}).")
            # Return empty array with correct feature dimension if possible
            n_features = data.shape[1] if data.ndim == 2 else 0
            return np.empty((0, sequence_length, n_features))
        for i in range(n_samples_total - sequence_length + 1):
            sequences.append(data[i:(i + sequence_length)])
        if not sequences: return np.empty((0, sequence_length, data.shape[1]))
        return np.array(sequences)
    
    def _preprocess_and_create_sequences(self, input_data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Scales and windows input data (DataFrame or 3D NumPy), returning 3D NumPy sequences.
        """
        if self.scaler is None or self.sequence_length is None or self.model is None:
            raise RuntimeError("Model/scaler not ready or sequence length unknown. Call run() first.")

        n_features_expected = self.model.input_shape[-1]
        X: Optional[np.ndarray] = None

        if isinstance(input_data, pd.DataFrame):
            #print("Preprocessing DataFrame...") # Optional print
            if input_data.shape[1] != n_features_expected:
                raise ValueError(f"Input DataFrame has {input_data.shape[1]} features, model expects {n_features_expected}.")
            try:
                data_normalized = self.scaler.transform(input_data)
                X = self.__create_sequences(data_normalized, self.sequence_length)
            except Exception as e: raise RuntimeError(f"Failed to scale/sequence DataFrame: {e}") from e

        elif isinstance(input_data, np.ndarray):
            #print("Preprocessing NumPy array...") # Optional print
            if input_data.ndim != 3: raise ValueError(f"NumPy input must be 3D, got {input_data.ndim}D.")
            if input_data.shape[1] != self.sequence_length: raise ValueError(f"NumPy seq length {input_data.shape[1]} != expected {self.sequence_length}.")
            if input_data.shape[2] != n_features_expected: raise ValueError(f"NumPy features {input_data.shape[2]} != expected {n_features_expected}.")

            X_input_3d = input_data
            n_samples, seq_len, n_feat = X_input_3d.shape
            try:
                X_reshaped_2d = X_input_3d.reshape(-1, n_feat)
                X_scaled_2d = self.scaler.transform(X_reshaped_2d)
                X = X_scaled_2d.reshape(n_samples, seq_len, n_feat)
            except Exception as e: raise RuntimeError(f"Failed to scale 3D NumPy input: {e}") from e
        else:
            raise TypeError("Input must be a pandas DataFrame or a 3D NumPy array.")

        if X is None or X.size == 0:
            warnings.warn("No sequences created from input data.", RuntimeWarning)
            return np.empty((0, self.sequence_length, n_features_expected))

        # Ensure float32 for model prediction
        if X.dtype != np.float32:
            try: X = X.astype(np.float32)
            except ValueError: warnings.warn("Could not cast sequences to float32.", RuntimeWarning)

        return X


    # --- Method to get anomaly score ---
    def get_anomaly_score(self, detection_data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Calculates reconstruction error for input data using the trained autoencoder.
        Higher error indicates a higher likelihood of anomaly.
        Accepts DataFrame (features) or 3D NumPy (pre-windowed sequences).

        Returns:
            np.ndarray: 1D array of reconstruction errors per sequence (shape: (n_sequences,)).
        """
        #print("Calculating anomaly scores (reconstruction error)...") # Optional print
        # Preprocess input data into scaled 3D sequences
        X = self._preprocess_and_create_sequences(detection_data)

        if X.size == 0: return np.array([]) # No sequences to score

        # Get reconstruction from autoencoder
        #print(f"Predicting reconstructions for {X.shape[0]} sequences...") # Optional print
        try:
            reconstructed = self.model.predict(X)
        except Exception as e:
            raise RuntimeError(f"Model prediction failed during scoring. Input shape: {X.shape}. Error: {e}") from e

        # Handle shape mismatch (optional, but good practice)
        if X.shape != reconstructed.shape:
             warnings.warn(f"Shape mismatch input {X.shape} vs reconstruction {reconstructed.shape}.", RuntimeWarning)
             min_samples = min(X.shape[0], reconstructed.shape[0])
             if min_samples == 0: return np.array([])
             # Calculate error only on matching samples
             reconstruction_error = np.mean(np.square(X[:min_samples] - reconstructed[:min_samples]), axis=(1, 2))
        else:
            # Calculate reconstruction error (MSE per sequence)
             reconstruction_error = np.mean(np.square(X - reconstructed), axis=(1, 2))

        #print(f"Calculated {len(reconstruction_error)} scores.") # Optional print
        return reconstruction_error # Return 1D scores

    # Detects anomalies and returns a list of boolean values
    def detect(self, detection_data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Detects anomalies by comparing reconstruction error scores to the threshold.
        Accepts DataFrame (features) or 3D NumPy (pre-windowed sequences).

        Returns:
             np.ndarray: A 1D boolean array (True=Anomaly), shape (n_sequences,).
        """
        if self.threshold is None:
             raise RuntimeError("Threshold not set. Call run() first.")

        # Get the reconstruction error scores using the new method
        scores = self.get_anomaly_score(detection_data)

        if scores.size == 0: return np.array([], dtype=bool)

        # Compare scores to threshold (higher error = anomaly for reconstruction)
        anomalies = scores > self.threshold
        print(f"Detected {np.sum(anomalies)} anomalies using threshold {self.threshold:.6f}.")
        return np.array(anomalies) # Return 1D boolean array
    