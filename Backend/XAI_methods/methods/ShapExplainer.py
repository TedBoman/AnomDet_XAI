import shap
import numpy as np
import pandas as pd # Keep import in case internal handling requires it later
from typing import Any, Union, Dict, List, Optional
import warnings

# Import the base class API
from XAI_methods.explainer_method_api import ExplainerMethodAPI

class ShapExplainer(ExplainerMethodAPI):
    """
    SHAP implementation conforming to the ExplainerMethodAPI.

    Uses shap.KernelExplainer for model-agnostic explanations of time series models.
    Assumes the model expects input sequences (e.g., NumPy arrays of shape
    (n_samples, sequence_length, n_features)).
    """

    def __init__(self, model: Any, background_data: np.ndarray, **params: Any):
        print("Initializing ShapExplainer...")
        self.model = model
        self.mode = params.get('mode', 'regression').lower()
        self.shap_method = params.get('shap_method', 'kernel').lower()

        # Validate the selected SHAP method
        supported_methods = ['kernel', 'tree', 'linear', 'partition']
        if self.shap_method not in supported_methods:
             raise ValueError(f"Unsupported SHAP method: '{self.shap_method}'. Supported methods are: {supported_methods}")
        print(f"Selected SHAP method: '{self.shap_method}'")
        print(f"Prediction mode: '{self.mode}'")

        self.background_data = background_data

        # --- Define sequence shape attributes ---
        if background_data.ndim == 3:
             # Store the shape of the sequence dimensions (e.g., (seq_len, n_features))
            self._original_sequence_shape = background_data.shape[1:]
            # Calculate the number of flattened features
            self._num_flat_features = np.prod(self._original_sequence_shape)
            print(f"Derived sequence shape: {self._original_sequence_shape}, Flat features: {self._num_flat_features}")
        else:
             # Cannot proceed if background data is not 3D
             raise ValueError("Background data must be 3D to determine sequence shape.")

        # --- Step 1: Explicitly set the link based on mode ---
        if self.mode == 'classification':
            self.link_arg = "logit" # Use the string name
            print(f"Mode is '{self.mode}', setting link argument to string 'logit'")
        else: # regression
            self.link_arg = "identity" # Use the string name
            print(f"Mode is '{self.mode}', setting link argument to string 'identity'")

        # --- Step 2: Add debug prints ---
        print(f"--- Debug: Link object before KernelExplainer init ---")
        print(f"Using link object: {self.link_arg}")
        print(f"Type of link object: {type(self.link_arg)}")
        # Optional: Check if it seems like a function/callable object
        print(f"Is link callable? {callable(self.link_arg)}")
        # Optional: See base classes (might be complex/internal)
        try:
             print(f"Link object MRO (inheritance): {type(self.link_arg).mro()}")
        except Exception:
             print("Could not get MRO for link object.")
        print(f"--- End Debug ---")

        # --- Define Internal Prediction Function ---
        def _predict_fn_shap(data_2d: np.ndarray) -> np.ndarray:
                # Reshape from SHAP's 2D format to model's expected 3D format
                num_samples = data_2d.shape[0]
                try:
                    # data_reshaped_3d has shape (n_samples, seq_len, n_features)
                    data_reshaped_3d = data_2d.reshape((num_samples,) + self._original_sequence_shape)
                except ValueError as e:
                    raise ValueError(f"Error reshaping data for model prediction in SHAP wrapper. Input shape: {data_2d.shape}, Target shape: {(num_samples,) + self._original_sequence_shape}. Error: {e}") from e

                # Call the WRAPPER's predict method.
                # self.model is the ModelWrapperForXAI instance passed to ShapExplainer.__init__
                # The wrapper now handles the internal conversion to DataFrame for the original model.
                predictions = self.model.predict(data_reshaped_3d)

                # Ensure output is 2D (n_samples, n_outputs) as expected by KernelExplainer
                if predictions.ndim == 1:
                    return predictions[:, np.newaxis]
                elif predictions.ndim == 0: # Handle case where predict loop only ran for 1 sample
                    return np.array([[predictions.item()]])
                elif predictions.ndim == 2:
                    return predictions
                else:
                    # Handle unexpected output shapes from the wrapper
                    raise ValueError(f"Wrapper's prediction function returned unexpected shape: {predictions.shape}. Expected 1D or 2D.")

        print("Internal prediction function defined.")

        # --- Prepare Background Data Summary ---
        print("Preparing background data summary...")
        n_bg_samples = self.background_data.shape[0]
        # Use _num_flat_features which is now defined
        background_data_flat = self.background_data.reshape(n_bg_samples, self._num_flat_features)
        k_summary = min(50, n_bg_samples)

        background_summary_np: Optional[np.ndarray] = None

        # --- Conditional Explainer Initialization ---

        if self.shap_method == 'kernel':
            print("Initializing shap.KernelExplainer...")

            # Set link argument for KernelExplainer
            if self.mode == 'classification':
                self.link_arg = "logit"
                print(f"Mode is '{self.mode}', setting KernelExplainer link to 'logit'")
            else: # regression
                self.link_arg = "identity"
                print(f"Mode is '{self.mode}', setting KernelExplainer link to 'identity'")

            # Prepare background data summary for KernelExplainer (optional but common for speed)
            k_summary = min(50, n_bg_samples)
            background_summary_np: Optional[np.ndarray] = None

            if n_bg_samples > k_summary * 2:
                 print(f"Summarizing background data using shap.kmeans (k={k_summary})...")
                 try:
                    summary_object = shap.kmeans(background_data_flat, k_summary, round_values=False)
                    if hasattr(summary_object, 'data') and isinstance(summary_object.data, np.ndarray):
                        background_summary_np = summary_object.data
                        print(f"Extracted NumPy array from kmeans summary. Shape: {background_summary_np.shape}")
                    else:
                        warnings.warn("Could not extract NumPy data from shap.kmeans result structure. Falling back to raw data.", RuntimeWarning)
                        background_summary_np = background_data_flat
                 except Exception as kmeans_err:
                    print(f"WARNING: shap.kmeans failed ({kmeans_err}). Using raw background data instead. This might be slow.")
                    background_summary_np = background_data_flat
            else:
                print("Using raw background data for SHAP KernelExplainer (no summarization).")
                background_summary_np = background_data_flat

            print(f"Background data for KernelExplainer prepared. Shape: {background_summary_np.shape}")

            try:
                self._explainer = shap.KernelExplainer(
                    _predict_fn_shap,
                    background_summary_np,
                    link=self.link_arg
                 )
            except Exception as e:
                print(f"Error initializing shap.KernelExplainer: {e}")
                raise # Re-raise the exception after printing

        elif self.shap_method == 'tree':
            print("Initializing shap.TreeExplainer...")

            # TreeExplainer needs the raw model object
            # ASSUMPTION: The wrapper instance self.model exposes the raw model
            if not hasattr(self.model, '_model'):
                raise AttributeError(
                    "shap_method='tree' requires the 'model' wrapper instance "
                    "to have an attribute '_model' exposing the raw tree model."
                )
            raw_model = self.model.model.model

            # Set model_output argument for TreeExplainer
            if self.mode == 'classification':
                # 'probability' is common for tree models returning class probabilities
                self._shap_model_output = "raw"
                print(f"Mode is '{self.mode}', setting TreeExplainer model_output to 'probability'")
                # Optional: Check if the model output shape seems compatible with probabilities
                try:
                    # Predict on a single dummy instance from background data
                    dummy_instance = background_data_flat[0:1] # Needs 2D input for predict wrapper if used
                    # Need to reshape dummy_instance back to original shape for wrapper.predict call
                    dummy_prediction = self.model.predict(dummy_instance.reshape(1, *self._original_sequence_shape))
                    if dummy_prediction.ndim < 2 or dummy_prediction.shape[1] <= 1:
                        warnings.warn(
                            "Model prediction for classification mode returned shape "
                            f"{dummy_prediction.shape}. TreeExplainer with model_output='probability' "
                            "usually expects >1 output column (probabilities for each class). "
                            "This might lead to unexpected SHAP values.", RuntimeWarning
                        )
                except Exception as e:
                    warnings.warn(f"Could not check model prediction shape for classification mode compatibility with TreeExplainer: {e}", RuntimeWarning)

            else: # regression
                self._shap_model_output = "raw" # Or 'regression' depending on SHAP version/docs
                print(f"Mode is '{self.mode}', setting TreeExplainer model_output to 'raw'")

            # TreeExplainer can use the raw background data (flattened)
            print("Using raw background data for SHAP TreeExplainer.")
            try:
                self._explainer = shap.TreeExplainer(
                    raw_model,
                    data=background_data_flat, # Use the flattened background data
                    model_output=self._shap_model_output,
                    feature_perturbation='interventional'
                )
                # Optional: Check if the model is actually tree-based after init
                if not isinstance(self._explainer, shap.TreeExplainer):
                    warnings.warn(
                        f"Initialized SHAP explainer is not a TreeExplainer ({type(self._explainer)}). "
                        "The underlying model might not be a supported tree model, even though 'tree' method was requested.",
                        RuntimeWarning
                    )

            except Exception as e:
                print(f"Error initializing shap.TreeExplainer. Ensure the 'model' wrapper's 'underlying_model' attribute "
                    f"is a valid tree-based model supported by SHAP (e.g., LightGBM, XGBoost, sklearn tree/forest). Error: {e}")
                raise # Re-raise the exception after printing

        elif self.shap_method == 'linear':
            print("Initializing shap.LinearExplainer...")

            # LinearExplainer needs the raw model object
            # ASSUMPTION: The wrapper instance self.model exposes the raw model
            if not hasattr(self.model, '_model'):
                raise AttributeError(
                    "shap_method='linear' requires the 'model' wrapper instance "
                    "to have an attribute 'underlying_model' exposing the raw linear model."
                )
            raw_model = self.model.model.model

            # LinearExplainer can use the raw background data (flattened)
            # The data argument can help infer coefficients or set expectation
            print("Using raw background data for SHAP LinearExplainer.")
            try:
                self._explainer = shap.LinearExplainer(
                    raw_model,
                    background_data_flat # Use the flattened background data
                    # LinearExplainer doesn't use link/model_output in the same way
                )
                # Optional: Check if the model is actually linear after init
                if not isinstance(self._explainer, shap.LinearExplainer):
                    warnings.warn(
                        f"Initialized SHAP explainer is not a LinearExplainer ({type(self._explainer)}). "
                        "The underlying model might not be a supported linear model, even though 'linear' method was requested.",
                        RuntimeWarning
                    )

            except Exception as e:
                print(f"Error initializing shap.LinearExplainer. Ensure the 'model' wrapper's 'underlying_model' attribute "
                    f"is a valid linear model supported by SHAP (e.g., sklearn linear models). Error: {e}")
                raise # Re-raise the exception after printing

        elif self.shap_method == 'partition':
            print("Initializing shap.PartitionExplainer...")

            # Set link argument for PartitionExplainer (similar to Kernel)
            if self.mode == 'classification':
                self.link_arg = "logit"
                print(f"Mode is '{self.mode}', setting PartitionExplainer link to 'logit'")
            else: # regression
                self.link_arg = "identity"
                print(f"Mode is '{self.mode}', setting PartitionExplainer link to 'identity'")

            # PartitionExplainer uses the raw background data (flattened) for its 'data' argument
            print("Using raw background data for SHAP PartitionExplainer.")
            try:
                self._explainer = shap.PartitionExplainer(
                    _predict_fn_shap, # Use the prediction function wrapper
                    background_data_flat, # Use the flattened background data
                    link=self.link_arg # PartitionExplainer supports link
                )
            except Exception as e:
                print(f"Error initializing shap.PartitionExplainer: {e}")
                raise # Re-raise the exception after printing


        print("ShapExplainer initialization complete.")

    @property
    def expected_value(self):
        """
        Returns the expected value (base value) from the underlying
        SHAP explainer instance. This is the average prediction
        over the background dataset. It might be a single value (regression)
        or an array (multi-class classification).
        """
        if hasattr(self, '_explainer') and hasattr(self._explainer, 'expected_value'):
            # For PartitionExplainer, expected_value might be None if data=None during init.
            # With data=background_data_flat, it should calculate it.
            return self._explainer.expected_value
        else:
            warnings.warn("Could not retrieve expected_value from internal SHAP explainer.", RuntimeWarning)
            # Consider calculating it manually if explainer doesn't provide it?
            # This is complex as it depends on the explainer type and output.
            # Returning None might be sufficient for now.
            return None # Or raise an appropriate error

    def explain(self,
                instances_to_explain: np.ndarray,
                **kwargs: Any) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Calculate SHAP values for the given instances using the initialized explainer
        and reshape results to the original data format if applicable.

        Args:
            instances_to_explain (np.ndarray): The instances to explain. Assumed
                to be a NumPy array preprocessed into the format expected by the
                model (e.g., shape (n_instances, ...) matching the background data).
            **kwargs (Any): Additional arguments passed directly to the
                underlying `shap_values` method of the selected explainer.
                Note: Arguments specific to one explainer (like 'nsamples'
                or 'l1_reg' for KernelExplainer) will only be relevant
                when that explainer is used.

        Returns:
            np.ndarray or list[np.ndarray]: Reshaped SHAP values.
                - For regression or binary classification (single output):
                  A NumPy array of shape (n_instances, ...) matching the original
                  instance feature shape.
                - For multi-class classification (multiple outputs):
                  A list of NumPy arrays, one for each class, each of shape
                  (n_instances, ...) matching the original instance feature shape.

        Raises:
            TypeError: If instances_to_explain is not a NumPy array.
            ValueError: If instance dimensions don't match background data or reshaping fails.
            Any exceptions raised by the underlying `shap_values` call.
        """
        print(f"ShapExplainer: Received {instances_to_explain.shape[0]} instances to explain using '{self.shap_method}' method.")

        if not isinstance(instances_to_explain, np.ndarray):
            raise TypeError(f"{type(self).__name__} expects instances_to_explain as a NumPy ndarray.")

        # Validate instance shape against background data shape (excluding sample dimension)
        if instances_to_explain.shape[1:] != self._original_sequence_shape:
            raise ValueError(
                f"Instance feature shape {instances_to_explain.shape[1:]} does not match "
                f"background data feature shape {self._original_sequence_shape}."
            )

        n_instances = instances_to_explain.shape[0]

        # Flatten instances for the explainer input (which expects 2D: samples x features)
        instances_flat = instances_to_explain.reshape(n_instances, self._num_flat_features)

        print(f"Calling SHAP {type(self._explainer).__name__}.shap_values...")

        # --- Filter kwargs based on the explainer type ---
        explainer_kwargs = kwargs.copy() # Start with a copy of all kwargs

        # These args are primarily for KernelExplainer and cause errors elsewhere
        if self.shap_method != 'kernel':
            explainer_kwargs.pop('nsamples', None) # Remove if present
            explainer_kwargs.pop('l1_reg', None)   # Remove if present

        try:
            # Pass all extra kwargs to shap_values. SHAP will usually ignore unknown ones.
            shap_values_flat = self._explainer.shap_values(
                instances_flat,
                **explainer_kwargs
            )
            print("SHAP calculation finished.")

            # --- Reshape the output back to the original feature shape ---
            # This applies if the original data was > 1D feature space (i.e., sequence, image, etc.)
            if self._original_sequence_shape is not None and len(self._original_sequence_shape) > 1:
                 target_shape = (n_instances,) + self._original_sequence_shape # (n_instances, ...)

                 if isinstance(shap_values_flat, list):
                      # Multi-class classification: Reshape each array in the list
                      reshaped_shap_values = []
                      for i, class_shap_values_flat in enumerate(shap_values_flat):
                           # Ensure the flat shape is compatible before reshaping
                           if class_shap_values_flat.shape[0] != n_instances or class_shap_values_flat.shape[1] != self._num_flat_features:
                                warnings.warn(f"SHAP values for class {i} have unexpected flat shape {class_shap_values_flat.shape}. Expected ({n_instances}, {self._num_flat_features}). Skipping reshape.", RuntimeWarning)
                                reshaped_shap_values.append(class_shap_values_flat) # Append raw if shape doesn't match
                                continue

                           try:
                                reshaped_shap_values.append(class_shap_values_flat.reshape(target_shape))
                           except ValueError as e:
                                raise ValueError(f"Failed to reshape SHAP values for class {i}. "
                                                 f"Flat shape: {class_shap_values_flat.shape}, Target shape: {target_shape}. Error: {e}") from e
                      print(f"Reshaped SHAP values to list (items: {len(reshaped_shap_values)}, item shape: {target_shape})")
                      return reshaped_shap_values
                 elif isinstance(shap_values_flat, np.ndarray):
                      # Regression or binary classification: Reshape the single array
                      # Ensure the flat shape is compatible before reshaping
                      if shap_values_flat.shape[0] != n_instances or shap_values_flat.shape[1] != self._num_flat_features:
                           warnings.warn(f"SHAP values have unexpected flat shape {shap_values_flat.shape}. Expected ({n_instances}, {self._num_flat_features}). Skipping reshape.", RuntimeWarning)
                           return shap_values_flat # Return raw if shape doesn't match

                      try:
                           reshaped_shap_values = shap_values_flat.reshape(target_shape)
                      except ValueError as e:
                           raise ValueError(f"Failed to reshape SHAP values. "
                                            f"Flat shape: {shap_values_flat.shape}, Target shape: {target_shape}. Error: {e}") from e
                      print(f"Reshaped SHAP values to array shape: {reshaped_shap_values.shape}")
                      return reshaped_shap_values
                 else:
                      warnings.warn(f"Unexpected type returned by shap_values: {type(shap_values_flat)}. Returning raw output.", RuntimeWarning)
                      return shap_values_flat
            else:
                 # If original data was 2D (samples x features), the flat output is already the desired shape
                 print("Returning 2D SHAP values (original data was 2D/tabular).")
                 return shap_values_flat


        except Exception as e:
            print(f"Error during SHAP values calculation: {e}")
            raise # Re-raise the exception after printing