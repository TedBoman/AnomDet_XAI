# File: dice_builder.py

from XAI_methods.methods import DiceExplainer
import dice_ml
from dice_ml.counterfactual_explanations import CounterfactualExplanations as DiceMlExplanation

import numpy as np
import pandas as pd
from typing import Any, Union, Dict, List, Optional
import warnings

# --- DiCE Builder Function (Simplified) ---
def build_dice_explainer(model: Any, background_data: np.ndarray, **params: Any) -> DiceExplainer:
    """
    Builds and initializes the DiceExplainer instance.

    Validates necessary parameters and instantiates the DiceExplainer class.

    Expects specific keys in params: 'feature_names', 'sequence_length',
    'background_outcomes', 'outcome_name', 'continuous_features_for_dice'.
    Optionally uses 'backend', 'dice_method'.
    """
    print("--- Starting DiCE Builder (Simplified) ---")

    # 1. --- Extract and Validate Necessary parameters ---
    # Extract parameters needed for validation here or just pass **params directly
    # to DiceExplainer and let its __init__ handle validation.
    # For clarity, let's validate essential ones here.
    base_feature_names = params.get('feature_names')
    sequence_length = params.get('sequence_length')
    background_outcomes = params.get('background_outcomes')
    outcome_name = params.get('outcome_name')
    continuous_feature_names = params.get('continuous_features_for_dice')

    # Basic validation checks (DiceExplainer.__init__ likely has more detailed ones)
    req_params_for_builder = ['feature_names', 'sequence_length', 'background_outcomes', 'outcome_name', 'continuous_features_for_dice']
    if not all(params.get(p) is not None for p in req_params_for_builder):
        missing = [p for p in req_params_for_builder if params.get(p) is None]
        raise ValueError(f"DiCE builder missing required parameters in params: {missing}")

    if not isinstance(background_data, np.ndarray) or background_data.ndim != 3:
         raise ValueError("Dice builder expects background_data as 3D NumPy array.")

    # Other minimal checks if needed...
    print("Builder: Parameters validated minimally.")

    # 2. --- Instantiate and Return DiceExplainer ---
    print("Builder: Instantiating DiceExplainer class...")
    try:
        # Pass the model, background_data, and all other params directly
        explainer_instance = DiceExplainer(
            model=model,
            background_data=background_data,
            **params # Pass all original params through
        )
        print("--- DiCE Builder Finished Successfully ---")
        return explainer_instance # Return the initialized DiceExplainer object
    except Exception as e:
        print(f"ERROR: Failed to instantiate DiceExplainer from builder: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError("DiCE Builder failed during DiceExplainer instantiation.") from e
