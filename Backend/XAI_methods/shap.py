import Backend.XAI_methods.methods.ShapExplainer as ShapExplainer
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

        if n_bg_samples > k_summary * 2:
            print(f"Summarizing background data using shap.kmeans (k={k_summary})...")
            try:
                # Added round_values=False which can sometimes help stability
                summary_object = ShapExplainer.kmeans(background_data_flat, k_summary, round_values=False)

                if hasattr(summary_object, 'data') and isinstance(summary_object.data, np.ndarray):
                    background_summary_np = summary_object.data
                    print(f"Extracted NumPy array from kmeans summary. Shape: {background_summary_np.shape}")
                else:
                    # Fallback if the structure is unexpected
                    warnings.warn("Could not extract NumPy data from shap.kmeans result structure. Falling back to raw data.", RuntimeWarning)
                    background_summary_np = background_data_flat # Use the original flattened data

            except Exception as kmeans_err:
                 print(f"WARNING: shap.kmeans failed ({kmeans_err}). Using raw background data instead. This might be slow.")
                 background_summary_np = background_data_flat
        else:
            print("Using raw background data for SHAP.")
            background_summary_np = background_data_flat
        print(f"Background data summary prepared. Shape: {background_summary_np.shape}")

        # --- Initialize KernelExplainer ---
        print("Initializing shap.KernelExplainer...")
        try:
            # Now _predict_fn_shap and background_summary_np are defined
            self._explainer = ShapExplainer.KernelExplainer(
                _predict_fn_shap,
                background_summary_np,
                link=self.link_arg
            )
        except TypeError as e:
            if "link function" in str(e) or "iml.Link" in str(e):
                print(f"\n--- ERROR: Passing link='{self.link_arg}' as string ALSO failed. ---")
                print("This confirms the issue is likely version incompatibility.")
                print("RECOMMENDATION: Update shap and numba libraries (`pip install --upgrade shap numba`)")
                raise TypeError(f"Invalid SHAP link argument ('{self.link_arg}'). Check SHAP/Numba versions.") from e
            else: raise # Re-raise other TypeErrors
        except Exception as e:
             print(f"An unexpected error occurred during KernelExplainer init: {e}")
             # Add details that might help diagnose
             print(f"  Predict function type: {type(_predict_fn_shap)}")
             print(f"  Background summary shape: {background_summary_np.shape if hasattr(self, 'background_summary') else 'N/A'}")
             print(f"  Link function: {self.link_arg}")
             raise
        print("ShapExplainer initialization complete.")

    @property
    def expected_value(self):
        """
        Returns the expected value (base value) from the underlying
        shap.KernelExplainer instance. This is the average prediction
        over the background dataset. It might be a single value (regression)
        or an array (multi-class classification).
        """
        if hasattr(self, '_explainer') and hasattr(self._explainer, 'expected_value'):
            return self._explainer.expected_value
        else:
            warnings.warn("Could not retrieve expected_value from internal SHAP explainer.", RuntimeWarning)
            return None # Or raise an appropriate error

    def explain(self,
                instances_to_explain: np.ndarray,
                **kwargs: Any) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Calculate SHAP values for the given instances and reshape to sequence format.

        Args:
            instances_to_explain (np.ndarray): The instances to explain. Assumed
                to be a NumPy array preprocessed into the format expected by the
                model (e.g., shape (n_instances, sequence_length, n_features)).
            **kwargs (Any): Additional arguments passed directly to the
                            `shap.KernelExplainer.shap_values` method.
                            Common arguments: nsamples, l1_reg.

        Returns:
            np.ndarray or list[np.ndarray]: Reshaped SHAP values.
                - For regression or binary classification (single output from predict fn):
                  A NumPy array of shape (n_instances, sequence_length, n_features).
                - For multi-class classification (multiple outputs from predict fn):
                  A list of NumPy arrays, one for each class, each of shape
                  (n_instances, sequence_length, n_features).

        Raises:
            TypeError: If instances_to_explain is not a NumPy array.
            ValueError: If instance dimensions don't match background data or reshaping fails.
            Any exceptions raised by the underlying `shap_values` call.
        """
        print(f"ShapExplainer: Received {instances_to_explain.shape[0]} instances to explain.")

        if not isinstance(instances_to_explain, np.ndarray):
            raise TypeError(f"{type(self).__name__} expects instances_to_explain as a NumPy ndarray.")

        # Validate instance shape against background data shape (excluding sample dimension)
        if instances_to_explain.shape[1:] != self._original_sequence_shape:
            raise ValueError(
                f"Instance sequence shape {instances_to_explain.shape[1:]} does not match "
                f"background data sequence shape {self._original_sequence_shape}."
            )

        n_instances = instances_to_explain.shape[0]
        # Flatten instances for the KernelExplainer input
        instances_flat = instances_to_explain.reshape(n_instances, self._num_flat_features)

        # Extract SHAP specific arguments from kwargs
        nsamples = kwargs.get('nsamples', 'auto')
        l1_reg = kwargs.get('l1_reg', 'auto')
        other_shap_kwargs = {k: v for k, v in kwargs.items() if k not in ['nsamples', 'l1_reg']}

        print(f"Calling SHAP KernelExplainer.shap_values (nsamples={nsamples}, l1_reg='{l1_reg}')...")
        try:
            # Get the potentially flattened shap values
            shap_values_flat = self._explainer.shap_values(
                instances_flat,
                nsamples=nsamples,
                l1_reg=l1_reg,
                **other_shap_kwargs
            )
            print("SHAP calculation finished. Reshaping results...")

            # --- Reshape the output ---
            target_shape = (n_instances,) + self._original_sequence_shape # (n_instances, seq_len, n_features)

            if isinstance(shap_values_flat, list):
                # Multi-class classification: Reshape each array in the list
                reshaped_shap_values = []
                for i, class_shap_values_flat in enumerate(shap_values_flat):
                    try:
                        reshaped_shap_values.append(class_shap_values_flat.reshape(target_shape))
                    except ValueError as e:
                        raise ValueError(f"Failed to reshape SHAP values for class {i}. "
                                         f"Flat shape: {class_shap_values_flat.shape}, Target shape: {target_shape}. Error: {e}") from e
                print(f"Reshaped SHAP values to list (items: {len(reshaped_shap_values)}, item shape: {target_shape})")
                return reshaped_shap_values
            elif isinstance(shap_values_flat, np.ndarray):
                # Regression or binary classification: Reshape the single array
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

        except Exception as e:
            print(f"Error during SHAP values calculation or reshaping: {e}")
            raise