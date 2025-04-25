# File: dice_explainer.py

import dice_ml
import numpy as np
import pandas as pd
from typing import Any, Union, Dict, List, Optional
import warnings

# Import the base class API
try:
    from explainer_method_api import ExplainerMethodAPI
except ImportError:
    import abc
    class ExplainerMethodAPI(abc.ABC):
        @abc.abstractmethod
        def __init__(self, model: Any, background_data: np.ndarray, **params: Any): raise NotImplementedError
        @abc.abstractmethod
        def explain(self, instances_to_explain: np.ndarray, **kwargs: Any) -> Any: raise NotImplementedError


class DiceExplainer(ExplainerMethodAPI):
    """
    DiCE counterfactual implementation conforming to the ExplainerMethodAPI.

    Adapts dice-ml for time series by flattening sequences.
    Requires additional parameters during initialization beyond the standard API.

    NOTE: Counterfactuals generated on flattened sequences may be less interpretable
          than methods designed specifically for time series. Performance can also be slow.
    """

    def __init__(self, model: Any, background_data: np.ndarray, **params: Any):
        """
        Initializes the DiceExplainer.

        Args:
            model (Any): The trained model instance (or wrapper) with a `predict_proba`
                         method expecting 3D NumPy input (batch, seq, feat) and
                         returning class probabilities (batch, n_classes).
            background_data (np.ndarray): 3D NumPy array (samples, seq_len, features)
                                          of background feature data.
            **params (Any): Additional parameters required by DiCE:
                - REQUIRED:
                    - feature_names (List[str]): List of BASE feature names.
                    - sequence_length (int): Sequence length dimension.
                    - background_outcomes (np.ndarray): 1D NumPy array of outcome labels
                                                      corresponding to background_data samples.
                    - outcome_name (str): Name for the outcome/target column.
                    - continuous_feature_names (List[str]): List of BASE feature names
                                                           that are continuous.
                - OPTIONAL (passed to dice_ml):
                    - backend (str): e.g., 'sklearn', 'TF2'. Default 'sklearn'.
        """
        print("Initializing DiceExplainer...")
        self.model = model # This should be the wrapper instance
        self.mode = 'classification' # DiCE typically requires classification for counterfactuals

        # --- Get required parameters specific to DiCE ---
        self.base_feature_names = params.get('feature_names')
        self.sequence_length = params.get('sequence_length')
        self.background_outcomes = params.get('background_outcomes')
        self.outcome_name = params.get('outcome_name')
        self.continuous_feature_names = params.get('continuous_feature_names')
        self.backend = params.get('backend', 'sklearn') # Default backend
        self.features_to_vary = params.get('features_to_vary', [])

        if not all([self.base_feature_names, self.sequence_length, self.background_outcomes is not None, self.outcome_name, self.continuous_feature_names is not None]):
             raise ValueError("DiceExplainer requires 'feature_names', 'sequence_length', 'background_outcomes', 'outcome_name', and 'continuous_feature_names' in params.")
        if not isinstance(self.base_feature_names, list) or not isinstance(self.continuous_feature_names, list):
             raise TypeError("'feature_names' and 'continuous_feature_names' must be lists.")
        if not isinstance(self.background_outcomes, np.ndarray):
             raise TypeError("'background_outcomes' must be a NumPy array.")
        # --- End Required Params ---


        # --- Validate background data & outcomes ---
        if not isinstance(background_data, np.ndarray) or background_data.ndim != 3:
            raise ValueError("DiceExplainer expects background_data as 3D NumPy array.")
        if background_data.shape[0] != len(self.background_outcomes):
             raise ValueError(f"Background data samples ({background_data.shape[0]}) != background outcomes ({len(self.background_outcomes)}).")
        if background_data.shape[1] != self.sequence_length:
             raise ValueError(f"Background data seq length ({background_data.shape[1]}) != expected ({self.sequence_length}).")
        n_features = background_data.shape[2]
        if n_features != len(self.base_feature_names):
            raise ValueError(f"Background data features ({n_features}) != length of feature_names ({len(self.base_feature_names)}).")

        self._original_sequence_shape = background_data.shape[1:]
        self._num_flat_features = np.prod(self._original_sequence_shape)

        # --- Prepare Data for dice_ml.Data ---
        print("Preparing data for DiCE (flattening sequences)...")
        # Flatten background features
        background_flat_2d = background_data.reshape(-1, self._num_flat_features)
        # Create flattened feature names
        self.flat_feature_names = [f"{feat}_t-{i}" for i in range(self.sequence_length -1, -1, -1) for feat in self.base_feature_names]
        # Create DataFrame required by dice_ml.Data
        dice_training_df = pd.DataFrame(background_flat_2d, columns=self.flat_feature_names)
        dice_training_df[self.outcome_name] = self.background_outcomes # Add outcome column
        print(f"Created DataFrame for DiCE Data object, shape: {dice_training_df.shape}")

        # Identify flattened continuous feature names
        flat_continuous_list = [f"{feat}_t-{i}" for i in range(self.sequence_length -1, -1, -1) for feat in self.continuous_feature_names]
        # Verify they exist in the flat names (should always be true)
        flat_continuous_list = [f for f in flat_continuous_list if f in self.flat_feature_names]
        print(f"Identified {len(flat_continuous_list)} flattened continuous features.")


        # --- Create dice_ml.Data object ---
        try:
            self.dice_data = dice_ml.Data(
                dataframe=dice_training_df,
                continuous_features=flat_continuous_list,
                outcome_name=self.outcome_name
            )
            print("dice_ml.Data object created.")
        except Exception as e:
            raise RuntimeError(f"Failed to create dice_ml.Data object: {e}") from e


        # --- Define Prediction Function for DiCE Model ---
        # Takes 2D FLATENED NumPy array (k, seq*feat) -> Returns 2D Probabilities (k, n_classes)
        def _predict_fn_dice(data_flat_2d: np.ndarray) -> np.ndarray:
            num_dice_samples = data_flat_2d.shape[0]
            try:
                # Reshape flat perturbations to 3D sequences for the model wrapper
                data_reshaped_3d = data_flat_2d.reshape((num_dice_samples,) + self._original_sequence_shape)
            except ValueError as e:
                raise ValueError(f"DiCE Predict Fn: Error reshaping flat data {data_flat_2d.shape} to sequence shape {(num_dice_samples,) + self._original_sequence_shape}. Error: {e}") from e

            # Use the predict_proba method from the wrapped model
            predict_proba_func = getattr(self.model, 'predict_proba', None)
            if predict_proba_func is None or not callable(predict_proba_func):
                raise AttributeError("DiCE requires the wrapped model to have a callable 'predict_proba' method.")

            # Call the wrapper's predict_proba
            probabilities = predict_proba_func(data_reshaped_3d) # Expects 3D, returns 2D (samples, classes)

            # Ensure output shape is suitable for DiCE: (samples, n_classes)
            if not isinstance(probabilities, np.ndarray) or probabilities.ndim != 2:
                 raise ValueError(f"DiCE Predict Fn: Wrapped model's predict_proba returned unexpected shape/type {probabilities.shape if hasattr(probabilities,'shape') else type(probabilities)}. Expected 2D (samples, n_classes).")
            # Optional: Add check for probability sum == 1?

            return probabilities # Return 2D probabilities

        self.predict_fn_for_dice = _predict_fn_dice
        print("DiCE prediction function defined.")


        # --- Create dice_ml.Model object ---
        try:
            # Pass the predict function wrapper, not the model object itself for model-agnostic use
            self.dice_model = dice_ml.Model(
                 func=self.predict_fn_for_dice,
                 backend=self.backend, # 'sklearn', 'TF2', etc.
                 model_type=self.mode, # 'classification' or 'regression'
                 features_to_vary=self.features_to_vary,
            )
            print(f"dice_ml.Model object created (backend: {self.backend}, type: {self.mode}).")
        except Exception as e:
             raise RuntimeError(f"Failed to create dice_ml.Model object: {e}") from e

        # --- Create dice_ml.Dice explainer object ---
        try:
            # Method: 'random', 'kdtree', 'genetic'
            dice_method = params.get('dice_method', 'genetic')
            self._explainer = dice_ml.Dice(self.dice_data, self.dice_model, method=dice_method)
            print(f"dice_ml.Dice object created (method: {dice_method}).")
        except Exception as e:
             raise RuntimeError(f"Failed to create dice_ml.Dice object: {e}") from e

        print("DiceExplainer initialization complete.")


    @property
    def expected_value(self):
        """ DiCE does not compute a global expected value. """
        warnings.warn("DiCE does not have a concept of 'expected_value'.", UserWarning)
        return None


    def explain(self,
                instances_to_explain: np.ndarray, # Expects 3D (n, seq, feat)
                **kwargs: Any) -> dice_ml.explanation.Explanation:
        """
        Generates diverse counterfactual explanations for given instance(s).

        Args:
            instances_to_explain (np.ndarray): One or more sequences in 3D NumPy
                format (n_instances, sequence_length, n_features).
            **kwargs (Any): Required arguments for DiCE's generate_counterfactuals:
                - total_CFs (int): Number of counterfactuals to generate per instance.
                - desired_class (int or "opposite"): Target outcome for the counterfactuals.
                                                    Defaults to "opposite".
                - Other optional args for generate_counterfactuals (e.g., features_to_vary).

        Returns:
            dice_ml.explanation.Explanation: The DiCE Explanation object containing
                                             original instances and counterfactuals.

        Raises:
            TypeError: If input is not a NumPy array.
            ValueError: If input is not 3D/empty, or required kwargs are missing.
        """
        print("DiceExplainer: Received explain request.")
        if not isinstance(instances_to_explain, np.ndarray):
            raise TypeError(f"{type(self).__name__} expects instances_to_explain as a NumPy ndarray.")
        if instances_to_explain.ndim != 3:
             raise ValueError(f"DiceExplainer expects 3D NumPy input (instances, seq, feat), got {instances_to_explain.ndim}D.")
        if instances_to_explain.shape[0] == 0:
             raise ValueError("Input instances_to_explain is empty.")
        # Validate shape against internal params
        if instances_to_explain.shape[1:] != self._original_sequence_shape:
            raise ValueError(f"Instance sequence shape {instances_to_explain.shape[1:]} doesn't match expected {self._original_sequence_shape}.")

        # --- Extract DiCE arguments from kwargs ---
        total_CFs = kwargs.get('total_CFs')
        desired_class = kwargs.get('desired_class', "opposite") # Default to opposite
        other_dice_kwargs = {k: v for k, v in kwargs.items() if k not in ['total_CFs', 'desired_class']}

        if total_CFs is None:
            raise ValueError("DiceExplainer.explain requires 'total_CFs' (int) in kwargs.")
        try:
            total_CFs = int(total_CFs)
            if total_CFs < 1: raise ValueError()
        except:
            raise ValueError("'total_CFs' must be a positive integer.")
        # --- End DiCE args ---

        # --- Prepare Query Instances ---
        # Flatten instances to 2D and create DataFrame
        n_instances = instances_to_explain.shape[0]
        instances_flat_2d = instances_to_explain.reshape(n_instances, self._num_flat_features)
        query_df = pd.DataFrame(instances_flat_2d, columns=self.flat_feature_names)
        # Remove outcome column if accidentally present (shouldn't be)
        query_df = query_df.drop(columns=[self.outcome_name], errors='ignore')
        print(f"Prepared query DataFrame for DiCE, shape: {query_df.shape}")


        # --- Generate Counterfactuals ---
        print(f"Calling DiCE generate_counterfactuals (total_CFs={total_CFs}, desired_class='{desired_class}')...")
        try:
            dice_explanation = self._explainer.generate_counterfactuals(
                query_instances=query_df,
                total_CFs=total_CFs,
                desired_class=desired_class,
                **other_dice_kwargs
            )
            # Potential issue: If query_df has >1 row, how does visualization work?
            # dice_exp.visualize_as_dataframe() handles multiple query instances.
            print("DiCE counterfactual generation finished.")
            return dice_explanation

        except Exception as e:
            print(f"Error during DiCE generate_counterfactuals call: {e}")
            import traceback
            traceback.print_exc() # Print full traceback
            raise RuntimeError("DiCE counterfactual generation failed.") from e