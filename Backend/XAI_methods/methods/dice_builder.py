# File: dice_builder.py

import dice_ml
from dice_ml.counterfactual_explanations import CounterfactualExplanations as DiceMlExplanation

import numpy as np
import pandas as pd
from typing import Any, Union, Dict, List, Optional
import warnings

# Import the API base class that the final wrapper must implement
try:
    from explainer_method_api import ExplainerMethodAPI
except ImportError:
    import abc
    class ExplainerMethodAPI(abc.ABC): # Dummy for structure
        @abc.abstractmethod
        def __init__(self, model: Any, background_data: np.ndarray, **params: Any): raise NotImplementedError
        @abc.abstractmethod
        def explain(self, instances_to_explain: np.ndarray, **kwargs: Any) -> Any: raise NotImplementedError


# --- Wrapper Class returned by the builder ---
class DiceExplainerWrapper(ExplainerMethodAPI):
    """ Thin wrapper around an initialized dice_ml.Dice object """

    # No __init__ needed here matching ExplainerMethodAPI signature
    # This wrapper is *created* by the builder fully initialized.
    def __init__(self):
         self.dice_ml_explainer: Optional[dice_ml.Dice] = None
         self.flat_feature_names: Optional[List[str]] = None
         self.outcome_name: Optional[str] = None
         self._original_sequence_shape: Optional[tuple] = None
         self._num_flat_features: Optional[int] = None

    @property
    def expected_value(self): return None # Not applicable to DiCE

    def explain(self, instances_to_explain: np.ndarray, **kwargs: Any) -> DiceMlExplanation:
        """ Generates counterfactuals using the pre-initialized DiCE explainer. """
        print("DiceExplainerWrapper: Received explain request.")
        # --- Input Checks ---
        if not all([self.dice_ml_explainer, self.flat_feature_names, self._original_sequence_shape, self._num_flat_features]):
             raise RuntimeError("DiceExplainerWrapper not properly initialized.")
        if not isinstance(instances_to_explain, np.ndarray) or instances_to_explain.ndim != 3:
             raise ValueError("DiceExplainerWrapper expects 3D NumPy input (instances, seq, feat).")
        if instances_to_explain.shape[1:] != self._original_sequence_shape:
             raise ValueError(f"Instance shape {instances_to_explain.shape[1:]} != expected {self._original_sequence_shape}.")
        if instances_to_explain.shape[0] == 0:
             raise ValueError("Cannot explain empty instances array.")

        # --- Extract DiCE explain() arguments ---
        total_CFs = kwargs.get('total_CFs')
        desired_class = kwargs.get('desired_class', "opposite")
        # Get the list of BASE feature names intended to be varied
        features_to_vary_base = kwargs.get('features_to_vary', []) # Expects a list from settings

        # --- START: Transform base feature names to flattened feature names ---
        features_to_vary_final: Union[List[str], str]
        if isinstance(features_to_vary_base, list) and features_to_vary_base:
            print(f"Transforming base features_to_vary list: {features_to_vary_base}")
            features_to_vary_flat_set = set()
            sequence_length = self._original_sequence_shape[0] # Get sequence length from stored shape

            for base_feat in features_to_vary_base:
                # Generate all flattened names for this base feature across time steps
                for i in range(sequence_length - 1, -1, -1): # Match naming convention t-N..t-0
                    flat_name = f"{base_feat}_t-{i}"
                    # IMPORTANT: Check if the generated flat name actually exists in the list known to the wrapper
                    if flat_name in self.flat_feature_names:
                        features_to_vary_flat_set.add(flat_name)
                    else:
                        # This warning helps catch typos or mismatches in base feature names vs known flat names
                        print(f"Warning: Generated flat name '{flat_name}' for base feature '{base_feat}' not found in known flat features.")

            if not features_to_vary_flat_set:
                 print("Warning: Transformation resulted in empty set of flattened features to vary. DiCE might default to varying all or fail.")
                 features_to_vary_final = [] # Pass empty list
            else:
                 features_to_vary_final = sorted(list(features_to_vary_flat_set)) # Convert set to sorted list
                 print(f"Transformed to {len(features_to_vary_final)} flattened features_to_vary (first 10): {features_to_vary_final[:10]}...")

        elif isinstance(features_to_vary_base, str) and features_to_vary_base.lower() == 'all':
             features_to_vary_final = 'all' # Pass 'all' string directly if specified
             print("Using 'all' features_to_vary.")
        else: # Default case: Empty list passed or unexpected type
             print("No specific features_to_vary provided or list is empty. Passing empty list to DiCE (may default to 'all').")
             features_to_vary_final = [] # Pass empty list to DiCE

        # --- END: Transform features_to_vary ---

        # Collect other kwargs, EXCLUDING the original 'features_to_vary' key if it existed
        other_dice_kwargs = {k: v for k, v in kwargs.items() if k not in ['total_CFs', 'desired_class', 'features_to_vary']}
        # Add the CORRECTLY FORMATTED features_to_vary to the kwargs going to DiCE
        other_dice_kwargs['features_to_vary'] = features_to_vary_final
        
        if total_CFs is None: raise ValueError("explain() requires 'total_CFs' (int) in kwargs for DiCE.")
        try: total_CFs = int(total_CFs); assert total_CFs >= 1
        except: raise ValueError("'total_CFs' must be a positive integer.")
        # --- End DiCE Args ---

        # --- Prepare Query Instances (Flatten to 2D DataFrame) ---
        n_instances = instances_to_explain.shape[0]
        instances_flat_2d = instances_to_explain.reshape(n_instances, self._num_flat_features)
        query_df = pd.DataFrame(instances_flat_2d, columns=self.flat_feature_names)
        query_df = query_df.drop(columns=[self.outcome_name], errors='ignore') # Ensure outcome not present
        print(f"Prepared query DataFrame for DiCE, shape: {query_df.shape}")

        # --- Generate Counterfactuals ---
        print(f"Calling DiCE generate_counterfactuals (total_CFs={total_CFs}, desired_class='{desired_class}')...")
        try:
            dice_explanation = self.dice_ml_explainer.generate_counterfactuals(
                query_instances=query_df,
                total_CFs=total_CFs,
                desired_class=desired_class,
                **other_dice_kwargs
            )
            print("DiCE counterfactual generation finished.")
            return dice_explanation
        except Exception as e:
            print(f"Error during DiCE generate_counterfactuals call: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError("DiCE counterfactual generation failed.") from e

# --- DiCE Builder Function ---
def build_dice_explainer(model: Any, background_data: np.ndarray, **params: Any) -> DiceExplainerWrapper:
    """
    Builds and initializes DiCE components using provided context and returns
    an API-compliant wrapper object.

    Expects specific keys in params: 'mode', 'feature_names', 'sequence_length',
    'training_df_for_dice', 'outcome_name_for_dice', 'continuous_features_for_dice'.
    """
    print("--- Starting DiCE Builder ---")

    # 1. --- Extract and Validate ALL necessary parameters ---
    mode = params.get('mode')
    base_feature_names = params.get('feature_names') # List of original feature names
    sequence_length = params.get('sequence_length')
    training_df = params.get('training_df_for_dice') # DF with features + outcome
    outcome_name = params.get('outcome_name_for_dice')
    continuous_feature_names = params.get('continuous_features_for_dice') # List of original continuous features
    backend = params.get('backend', 'sklearn') # Default backend if not passed
    dice_method = params.get('dice_method', 'genetic') # Default DiCE method

    print(f"DEBUG DiCE Builder: Retrieved base_feature_names: {base_feature_names}")
    print(f"DEBUG DiCE Builder: Length of base_feature_names: {len(base_feature_names) if base_feature_names else 'None'}")

    # --- Validation Checks ---
    if mode != 'classification': raise ValueError("DiCE builder requires mode='classification'.")
    req_params = ['feature_names', 'sequence_length', 'training_df_for_dice', 'outcome_name_for_dice', 'continuous_features_for_dice']
    if not all(params.get(p) is not None for p in req_params):
        missing = [p for p in req_params if params.get(p) is None]
        raise ValueError(f"DiCE builder missing required parameters in params: {missing}")
    if not isinstance(training_df, pd.DataFrame) or training_df.empty:
         raise ValueError("'training_df_for_dice' must be a non-empty pandas DataFrame.")
    if outcome_name not in training_df.columns:
         raise ValueError(f"Outcome column '{outcome_name}' not found in training_df_for_dice.")
    missing_feats = set(base_feature_names) - set(training_df.columns)
    if missing_feats: raise ValueError(f"Training DataFrame missing features: {missing_feats}")
    # ... (add other checks: sequence_length > 0, types etc.) ...
    n_features = len(base_feature_names)
    original_sequence_shape = (sequence_length, n_features)
    num_flat_features = sequence_length * n_features
    print("Builder: Parameters validated.")
    # --- End Validation ---


    # 2. --- Prepare Data for dice_ml.Data ---
    # Needs a DataFrame with FLATTENED features and the outcome column
    print("Builder: Preparing data for dice_ml.Data (requires windowing training DF)...")
    # Use the utility function to window the training features AND labels
    # Assuming dataframe_to_sequences can handle labels
    try:
        from utils import dataframe_to_sequences # Assumes this function is available
        # Select only needed columns (features + outcome)
        cols_for_windowing = base_feature_names + [outcome_name]
        training_df_subset = training_df[cols_for_windowing]

        # Generate 3D features and 1D labels
        # This creates the sequences that the *model wrapper* expects as input
        X_train_seq, y_train_aligned = dataframe_to_sequences(
            df=training_df_subset,
            sequence_length=sequence_length,
            feature_cols=base_feature_names,
            target_col=outcome_name
        )
        if X_train_seq.size == 0: raise ValueError("Windowing training data resulted in zero sequences.")

        # Flatten the 3D sequences for DiCE's DataFrame
        X_train_flat = X_train_seq.reshape(-1, num_flat_features)
        flat_feature_names = [f"{feat}_t-{i}" for i in range(sequence_length -1, -1, -1) for feat in base_feature_names]

        # Create the final DataFrame for DiCE
        dice_training_df = pd.DataFrame(X_train_flat, columns=flat_feature_names)
        dice_training_df[outcome_name] = y_train_aligned # Add the aligned outcome

        # Identify flattened continuous features
        flat_continuous_list = [f"{feat}_t-{i}" for i in range(sequence_length -1, -1, -1) for feat in continuous_feature_names]
        flat_continuous_list = [f for f in flat_continuous_list if f in flat_feature_names]
        print(f"Builder: Created DataFrame for DiCE data, shape: {dice_training_df.shape}")
        print(f"Builder: Identified {len(flat_continuous_list)} flattened continuous features.")

    except ImportError:
         raise ImportError("DiCE builder requires 'dataframe_to_sequences' utility function.")
    except Exception as e:
         raise RuntimeError(f"Builder: Error preparing windowed/flattened data for DiCE: {e}") from e


    # 3. --- Create dice_ml.Data object ---
    print("Builder: Creating dice_ml.Data...")
    try:
        dice_data = dice_ml.Data(
            dataframe=dice_training_df,
            continuous_features=flat_continuous_list,
            outcome_name=outcome_name
        )
    except Exception as e: raise RuntimeError(f"Builder: Failed dice_ml.Data init: {e}") from e


    # 4. --- Define Prediction Function Wrapper (Specific to this builder scope) ---
    # --- Define Prediction Function Wrapper (Specific to this builder scope) ---
    print("Builder: Defining DiCE prediction function wrapper...")
    # This function must accept DataFrame input matching dice_training_df (flattened features)
    # and return 2D NumPy probabilities (samples, classes)
    def _predict_proba_for_dice_model(data_flat_df: pd.DataFrame, **kwargs) -> np.ndarray:
        # DiCE provides perturbations as a DataFrame with flat feature names
        num_dice_samples = len(data_flat_df)
        if num_dice_samples == 0: return np.empty((0,2)) # Handle empty input

        # Check if we're dealing with probabilities already (fewer columns than expected)
        if data_flat_df.shape[1] < len(flat_feature_names):
            print(f"WARNING: Input appears to be probabilities already: shape {data_flat_df.shape}")
            # If input already has 2 columns (binary classification probabilities), return it as is
            if data_flat_df.shape[1] == 2:
                return data_flat_df.values
            # Otherwise, return a placeholder probability array
            return np.tile(np.array([[0.5, 0.5]]), (num_dice_samples, 1))

        # --- START DEBUG PRINTS ---
        print(f"\n--- DiCE Predict Fn Debug ---")
        print(f"Input DF shape: {data_flat_df.shape}")
        print(f"Input DF columns: {data_flat_df.columns.tolist()}") # See what columns DiCE actually provides
        print(f"Expected flat feature names (len={len(flat_feature_names)}): {flat_feature_names[:5]}...{flat_feature_names[-5:]}") # Check expected names
        # --- END DEBUG PRINTS ---

        try: # Add try block around data extraction and reshape
            data_flat_np = data_flat_df[flat_feature_names].values # Ensure correct order/features
            # --- MORE DEBUG PRINTS ---
            print(f"Selected data_flat_np shape: {data_flat_np.shape}")
            print(f"Target reshape shape tuple: {(num_dice_samples,) + original_sequence_shape}") # Check sequence_length and n_features here
            # --- END MORE DEBUG PRINTS ---

            # Reshape flat 2D NumPy -> 3D sequence NumPy for the model wrapper
            data_reshaped_3d = data_flat_np.reshape((num_dice_samples,) + original_sequence_shape)
            # --- FINAL SHAPE DEBUG PRINT ---
            print(f"Output data_reshaped_3d shape: {data_reshaped_3d.shape}")
            print(f"--- End DiCE Predict Fn Debug ---\n")
            # --- END FINAL SHAPE DEBUG PRINT ---

        except KeyError as e:
            print(f"ERROR: Key error during feature selection in DiCE predict fn. Missing columns? {e}")
            print(f"Available columns in data_flat_df: {data_flat_df.columns.tolist()}")
            raise ValueError(f"DiCE Predict Fn: DataFrame missing expected features. Error: {e}") from e
        except ValueError as e: # Catch potential reshape errors
            print(f"ERROR: Reshape error in DiCE predict fn. data_flat_np shape: {data_flat_np.shape}, target tuple: {(num_dice_samples,) + original_sequence_shape}")
            raise ValueError(f"DiCE Predict Fn: Reshape error: {e}") from e
        except Exception as e: # Catch any other unexpected errors
            print(f"ERROR: Unexpected error during data prep in DiCE predict fn: {e}")
            raise ValueError(f"DiCE Predict Fn: Unexpected data prep error: {e}") from e


        # Use the PREDICT_PROBA method of the model wrapper passed to the builder
        predict_proba_func = getattr(model, 'predict_proba', None)
        if predict_proba_func is None: raise AttributeError("DiCE requires wrapped model to have 'predict_proba'.")

        probabilities = predict_proba_func(data_reshaped_3d) # Call wrapper's predict_proba

        if not isinstance(probabilities, np.ndarray) or probabilities.ndim != 2:
                raise ValueError(f"DiCE Predict Fn: Wrapped predict_proba returned unexpected shape {probabilities.shape}. Expected (samples, classes).")
        return probabilities # Return 2D probabilities

    print("Builder: Prediction function defined.")


    # 5. --- Create dice_ml.Model object ---
    print("Builder: Creating dice_ml.Model...")
    # Create an instance of the placeholder
    actual_classes = np.array([0, 1])

    try:
        if mode == 'classification':
            dice_model_type = 'classifier'
        elif mode == 'regression':
            dice_model_type = 'regressor'
        else:
            # This case should ideally be caught by initial validation
            raise ValueError(f"Invalid mode '{mode}' encountered.") # Should not happen

        dice_model = dice_ml.Model(
            model=model,
            func=_predict_proba_for_dice_model, # Prediction function
            backend=backend,                    # Backend ('sklearn', 'TF2', 'PYT')
            model_type=dice_model_type,         # Pass 'classifier' or 'regressor'
        )
        print(f"dice_ml.Model object created (backend: {backend}, type: {dice_model_type}).")
    except Exception as e: raise RuntimeError(f"Builder: Failed dice_ml.Model init: {e}") from e


    # 6. --- Create dice_ml.Dice (The actual explainer) ---
    print(f"Builder: Creating dice_ml.Dice (method={dice_method})...")
    try:
        dice_ml_explainer = dice_ml.Dice(dice_data, dice_model, method=dice_method)
    except Exception as e: raise RuntimeError(f"Builder: Failed dice_ml.Dice init: {e}") from e


    # 7. --- Create and configure the API-compliant wrapper ---
    print("Builder: Creating DiceExplainerWrapper...")
    wrapper_instance = DiceExplainerWrapper()
    # Populate the wrapper with necessary objects/info
    wrapper_instance.dice_ml_explainer = dice_ml_explainer
    wrapper_instance.flat_feature_names = flat_feature_names
    wrapper_instance.outcome_name = outcome_name
    wrapper_instance._original_sequence_shape = original_sequence_shape
    wrapper_instance._num_flat_features = num_flat_features

    print("--- DiCE Builder Finished Successfully ---")
    return wrapper_instance # Return the simple wrapper