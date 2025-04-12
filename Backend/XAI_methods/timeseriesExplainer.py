import numpy as np
from typing import List, Any, Dict, Optional
from XAI_methods.xai_factory import xai_factory
from XAI_methods.explainer_method_api import ExplainerMethodAPI

class TimeSeriesExplainer:
    """
    Manages Time Series XAI operations.

    Holds shared resources (model, data) and uses an external method
    (`get_method`) to obtain and manage specific
    explainer objects (like SHAP or LIME wrappers) which perform the
    actual explanation tasks.
    """

    def __init__(self, model: Any, background_data: np.ndarray, feature_names: List[str], mode: str = 'regression'):
        """
        Initializes the TimeSeriesExplainer manager.

        Args:
            model: The trained machine learning model instance.
            background_data: Representative data sample (e.g., windowed sequences
                             in the format the model expects).
            feature_names: List of feature names corresponding to the last dimension
                           of the input data.
            mode: Prediction mode ('regression' or 'classification'). Affects how
                  some explainers might behave or interpret results.
        """
        if not hasattr(model, 'predict'):
             raise TypeError("Model must have a 'predict' method.")
        if mode == 'classification' and not hasattr(model, 'predict_proba'):
             print("Warning: Classification mode but model lacks 'predict_proba'. Some explainers might behave unexpectedly.")

        self._model = model
        # Optional: Add validation for background_data shape/type
        self._background_data = background_data
        self._feature_names = feature_names
        if mode not in ['regression', 'classification']:
            raise ValueError("Mode must be 'regression' or 'classification'")
        self._mode = mode

        # Cache to store initialized explainer objects returned by get_method
        self._explainer_cache: Dict[str, ExplainerMethodAPI] = {}
        print("TimeSeriesExplainer manager initialized.")

        print("TimeSeriesExplainer manager initialized.")

    # --- Public properties to access configuration ---
    @property
    def model(self) -> Any: return self._model
    @property
    def background_data(self) -> np.ndarray: return self._background_data
    @property
    def feature_names(self) -> List[str]: return self._feature_names
    @property
    def mode(self) -> str: return self._mode

    # --- Internal Helper ---
    def _get_or_initialize_explainer(self, method_name: str) -> ExplainerMethodAPI:
        """
        Retrieves a specific explainer object from the cache or initializes
        it by calling the external `get_method` factory function.
        """
        method_key = method_name.lower()
        if method_key not in self._explainer_cache:
            print(f"Initializing explainer for '{method_key}' via get_method...")
            try:
                # Call the external factory, passing all necessary context
                # get_method should handle which params are needed for which method
                explainer_object = xai_factory( # Call the corrected factory function
                    method_name=method_key,
                    ml_model=self._model,             # Pass the stored model
                    background_data=self._background_data, # Pass the stored background data
                    # Pass other necessary parameters as keyword arguments
                    mode=self._mode,
                    feature_names=self._feature_names # LIME might need this
                )
                self._explainer_cache[method_key] = explainer_object
                print(f"Initialized and cached explainer for '{method_key}'. Type: {type(explainer_object).__name__}")

            except (ValueError, RuntimeError, TypeError, ImportError) as e:
                print(f"Error initializing explainer '{method_key}': {e}")
                raise RuntimeError(f"Failed to get/initialize explainer '{method_key}'") from e
        else:
            print(f"Using cached explainer for '{method_key}'.")

        return self._explainer_cache[method_key]

    # --- Core Public Method ---
    def explain(self,
                instances_to_explain: np.ndarray, # Emphasize NumPy array input
                method_name: str,
                **kwargs: Any) -> Any:
        """
        Perform explanation using the specified method.

        Retrieves or initializes the appropriate specific explainer object
        (e.g., a SHAP wrapper) using the `get_method` factory and calls its
        `explain` method.

        Args:
            instances_to_explain (np.ndarray): The data instances (one or more)
                to explain. **MUST be a NumPy array** preprocessed into the
                3D sequence format expected by the model, typically
                `(n_instances, sequence_length, n_features)`. The conversion
                from original pandas DataFrames or other formats must happen
                *before* calling this method.
            method_name (str): Name of the explanation method (e.g., 'shap').
            **kwargs (Any): Additional keyword arguments passed directly to the
                            specific explainer object's `explain` method.

        Returns:
            Any: The result from the specific explainer's `explain` method.
                 For 'shap', the result is reshaped to match the input sequence
                 dimensions (e.g., (n_instances, sequence_length, n_features)).
                 Format for other methods depends on their implementation.

        Raises:
            RuntimeError: If the requested explainer cannot be initialized via `get_method`.
            TypeError: If `instances_to_explain` is not a NumPy array.
            ValueError: If instance shapes are inconsistent.
            Any exceptions raised by the specific explainer object's `explain` method.
        """
        print(f"\n--- TimeSeriesExplainer: Requesting explanation via method '{method_name}' ---")

        # Add explicit type check at the entry point for clarity
        if not isinstance(instances_to_explain, np.ndarray):
            raise TypeError("TimeSeriesExplainer.explain requires instances_to_explain to be a NumPy ndarray.")

        try:
            # Step 1: Get the specific explainer object (lazy initialization)
            explainer = self._get_or_initialize_explainer(method_name)

            # Step 2: Delegate to the specific object's explain method
            print(f"Calling '{method_name}' explainer's (.explain) method...")
            result = explainer.explain(instances_to_explain, **kwargs) # Pass the validated NumPy array

            print(f"--- TimeSeriesExplainer: Explanation finished for '{method_name}' ---")
            return result

        except Exception as e:
            print(f"Error during explanation process for method '{method_name}': {e}")
            raise

    def get_initialized_methods(self) -> List[str]:
        """Returns a list of method names for which explainers are currently initialized."""
        return list(self._explainer_cache.keys())