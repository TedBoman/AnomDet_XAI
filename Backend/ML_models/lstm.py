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
from typing import Union, List # Make sure to import Union
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
    def run(self, df, time_steps=10, epochs=10): # Changed default time_steps
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

    # Creates sequences - unchanged internally, but called with self.sequence_length
    def __create_sequences(self, data, time_steps):
        sequences = []
        # Correct loop range: stops when there aren't enough points left for a full sequence
        for i in range(len(data) - time_steps + 1):
            seq = data[i:i + time_steps]
            sequences.append(seq)
        return np.array(sequences)

    # Detects anomalies and returns a list of boolean values
    def detect(self, detection_data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Detects anomalies using the trained model. Accepts either a pandas DataFrame
        (raw features) or a pre-windowed 3D NumPy array.

        Args:
            detection_data (Union[pd.DataFrame, np.ndarray]):
                - If DataFrame: Contains features (shape: n_timesteps, n_features).
                                Must have the same features as training data.
                - If NumPy array: Contains pre-windowed sequences
                                  (shape: n_sequences, sequence_length, n_features).
                                  sequence_length and n_features must match the model.

        Returns:
            np.ndarray: A 1D boolean NumPy array indicating anomaly status for each
                        sequence evaluated. Length is n_sequences.
        """
        # 1. --- Prerequisite Checks ---
        if self.model is None or self.scaler is None or self.threshold is None or self.sequence_length is None:
            raise RuntimeError("Model is not trained or sequence length/scaler/threshold is not set. Call run() first.")

        X: Optional[np.ndarray] = None # To store the final 3D scaled sequences for the model
        n_features_expected = self.model.input_shape[-1] # Get expected features from model

        # 2. --- Input Type Handling & Preprocessing ---
        if isinstance(detection_data, pd.DataFrame):
            # --- DataFrame Path ---
            # Check feature count
            if detection_data.shape[1] != n_features_expected:
                raise ValueError(f"Input DataFrame has {detection_data.shape[1]} features, but model expects {n_features_expected}.")
            # Check for expected columns (optional but good)
            # if not all(col in detection_data.columns for col in self.feature_names): ...

            # Scale the 2D DataFrame features
            try:
                data_normalized = self.scaler.transform(detection_data)
            except ValueError as e:
                 raise ValueError(f"Could not scale DataFrame. Ensure it has the correct features ({n_features_expected}). Error: {e}") from e

            # Window the scaled 2D data into 3D sequences
            X = self.__create_sequences(data_normalized, self.sequence_length)

        elif isinstance(detection_data, np.ndarray):
            # --- NumPy Path ---
            # Validate NumPy array shape
            if detection_data.ndim != 3:
                raise ValueError(f"NumPy input must be 3D (samples/sequences, steps, features), got {detection_data.ndim}D.")
            if detection_data.shape[1] != self.sequence_length:
                raise ValueError(f"NumPy input sequence length ({detection_data.shape[1]}) does not match model sequence length ({self.sequence_length}).")
            if detection_data.shape[2] != n_features_expected:
                raise ValueError(f"NumPy input feature count ({detection_data.shape[2]}) does not match model feature count ({n_features_expected}).")

            # Input is already windowed (3D), just needs scaling
            X_input_3d = detection_data
            n_samples, seq_len, n_feat = X_input_3d.shape

            # Scale (requires reshape -> transform -> reshape back)
            try:
                # Reshape to 2D for scaler: (samples * steps, features)
                X_reshaped_2d = X_input_3d.reshape(-1, n_feat)
                # Apply scaler
                X_scaled_2d = self.scaler.transform(X_reshaped_2d)
                # Reshape back to 3D: (samples, steps, features)
                X = X_scaled_2d.reshape(n_samples, seq_len, n_feat)
            except Exception as e:
                # Catch potential errors during reshape or scaling
                raise RuntimeError(f"Failed to scale 3D NumPy input. Error: {e}") from e
        else:
             raise TypeError("Input 'detection_data' must be a pandas DataFrame or a 3D NumPy array.")

        # 3. --- Common Anomaly Detection Logic ---
        if X is None or X.size == 0:
            print("Warning: No valid sequences to process after preprocessing.")
            return np.array([], dtype=bool) # Return empty boolean array

        # Ensure X is float32 for many TF/Keras models
        if hasattr(self.model, 'predict') and X.dtype != np.float32:
             try:
                 X = X.astype(np.float32)
             except ValueError:
                 warnings.warn(f"Could not cast input sequences to float32. Model prediction might fail.", RuntimeWarning)

        print(f"Predicting reconstructions for {X.shape[0]} sequences...")
        try:
            reconstructed = self.model.predict(X) # Model predict expects 3D NumPy
        except Exception as e:
            print(f"ERROR during model.predict: {e}")
            # Consider how to handle prediction errors, maybe return empty/error indicator?
            raise RuntimeError(f"Model prediction failed. Input shape: {X.shape}") from e

        # Validate reconstruction shape
        if X.shape != reconstructed.shape:
             warnings.warn(f"Shape mismatch between input sequences {X.shape} and reconstructed sequences {reconstructed.shape}. Check model architecture or prediction output.", RuntimeWarning)
             # Attempt to proceed if only batch size differs (e.g., stateful prediction)
             min_samples = min(X.shape[0], reconstructed.shape[0])
             if min_samples == 0:
                 return np.array([], dtype=bool) # Cannot calculate error
             reconstruction_error = np.mean(np.square(X[:min_samples] - reconstructed[:min_samples]), axis=(1, 2))
        else:
            # Calculate reconstruction error (MSE per sequence)
             reconstruction_error = np.mean(np.square(X - reconstructed), axis=(1, 2))

        anomalies = reconstruction_error > self.threshold

        # Return 1D boolean numpy array as specified
        return np.array(anomalies) # Shape: (num_sequences,)