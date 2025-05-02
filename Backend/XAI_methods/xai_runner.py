# xai_runner.py
import sys
import warnings
import pandas as pd
import numpy as np
import os
import time
from typing import List, Optional, Dict, Any

# Custom / Third-party imports needed for XAI logic
from ML_models.model_wrapper import ModelWrapperForXAI
from XAI_methods.timeseriesExplainer import TimeSeriesExplainer
from XAI_methods import xai_visualizations as x
import utils as ut

# Constants or Configurations
OUTPUT_DIR = "/data"
MAX_BG_SAMPLES = 25000 # Example default, can be overridden

class XAIRunner:
    """
    Handles the execution of Explainable AI (XAI) methods for time series models.
    """
    def __init__(
        self,
        xai_params: List[Dict[str, Any]],
        model_wrapper: ModelWrapperForXAI,
        sequence_length: int,
        feature_columns: List[str],
        actual_label_col: str,
        continuous_features_list: List[str],
        job_name: str,
        mode: str = 'classification', # Default mode
        output_dir: str = OUTPUT_DIR
    ):
        """
        Initializes the XAIRunner.

        Args:
            xai_params (List[Dict[str, Any]]): List of configurations for each XAI method.
            model_wrapper (ModelWrapperForXAI): Wrapped model instance ready for XAI.
            sequence_length (int): The sequence length used by the model and for XAI.
            feature_columns (List[str]): List of base feature names.
            actual_label_col (str): Name of the column containing the true labels.
            continuous_features_list (List[str]): List of continuous feature names (for DiCE).
            job_name (str): Identifier for the current job (used for saving outputs).
            mode (str): Type of problem ('classification' or 'regression'). Defaults to 'classification'.
            output_dir (str): Directory to save XAI plots and results.
        """
        if not isinstance(xai_params, list):
            raise TypeError("xai_params must be a list of dictionaries.")
        if not model_wrapper:
            raise ValueError("model_wrapper cannot be None.")
        if not isinstance(sequence_length, int) or sequence_length <= 0:
            # Even non-sequential models need a sequence context for XAI framework
            raise ValueError("sequence_length must be a positive integer for XAI processing.")

        self.xai_params = xai_params
        self.model_wrapper = model_wrapper
        self.sequence_length = sequence_length
        self.feature_columns = feature_columns
        self.actual_label_col = actual_label_col
        self.continuous_features_list = continuous_features_list
        self.job_name = job_name
        self.mode = mode
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)
        print(f"XAIRunner initialized for job '{self.job_name}'. Output directory: '{self.output_dir}'")

    def run_explanations(
        self,
        training_features_df: pd.DataFrame, # Used for background data
        training_df_with_labels: pd.DataFrame, # Used for background outcomes and DiCE context
        data_source_for_explanation: pd.DataFrame # Data to explain (e.g., anomalies or all data)
    ):
        """
        Executes the configured XAI methods on the provided data.

        Args:
            training_features_df (pd.DataFrame): DataFrame with features used for training (for background data).
            training_df_with_labels (pd.DataFrame): Training DataFrame including the label column (for background outcomes).
            data_source_for_explanation (pd.DataFrame): DataFrame containing the instances to be explained.
        """
        print(f"\n--- Starting XAI Execution via XAIRunner for job '{self.job_name}' ---")
        print(f"Processing {len(self.xai_params)} XAI method(s)...")

        # --- Check Prerequisites ---
        # (Moved check for TimeSeriesExplainer and ut to the top-level import)
        if (TimeSeriesExplainer is None or ut.dataframe_to_sequences is None or
                x.process_and_plot_shap is None or x.process_and_plot_lime is None or x.process_and_plot_dice is None):
            print("XAI components not available (import failed). Skipping XAI.")
            return

        try:
            # --- 1. Prepare Common Background Data ---
            print(f"Preparing background data using sequence length {self.sequence_length}...")
            background_data_np = ut.dataframe_to_sequences(
                df=training_features_df,
                sequence_length=self.sequence_length,
                feature_cols=self.feature_columns
            )

            shap_method = 'kernel'
            # Determine SHAP method (needed before sampling background)
            # Assuming only one SHAP config for now, might need adjustment if multiple SHAP runs are allowed
            for config in self.xai_params:
                print(config)
                if config.get("method") == 'ShapExplainer':
                    settings = config.get('settings', {})
                    print(settings)
                    shap_method = settings.get('shap_method', 'kernel')
                    print(f"Using shap method: {shap_method} for background samples")

            max_bg_samples = MAX_BG_SAMPLES
            if shap_method == 'tree' and background_data_np.shape[0] > 0:
                # TreeSHAP typically uses all background data if feasible
                max_bg_samples = len(background_data_np)
                print(f"Using all {max_bg_samples} background samples for TreeSHAP.")
            elif len(background_data_np) > max_bg_samples:
                print(f"Sampling background data down from {len(background_data_np)} to {max_bg_samples} instances.")
                indices = np.random.choice(len(background_data_np), max_bg_samples, replace=False)
                background_data_np = background_data_np[indices]

            if background_data_np.size == 0:
                print("Warning: Background data generation resulted in empty array. Skipping XAI.")
                return # Stop if background fails

            # --- Extract Corresponding 1D Background Labels ---
            num_sequences = background_data_np.shape[0]
            end_indices = [i + self.sequence_length - 1 for i in range(num_sequences)] # Original indices based on full training set
            # Adjust indices if background data was sampled
            if len(background_data_np) < len(training_features_df) - self.sequence_length + 1:
                # Need to map sampled indices back or recalculate valid indices carefully.
                # Simplest robust way: re-calculate based on sampled indices if 'indices' variable exists
                if 'indices' in locals():
                    print("Adjusting label indices based on sampled background data...")
                    # The 'indices' array contains the STARTING index of each sampled sequence in the original training_features_df
                    end_indices = [start_idx + self.sequence_length - 1 for start_idx in indices]
                else: # Should not happen if sampling occurred, but as fallback:
                    warnings.warn("Background data was sampled but indices mapping is unclear. Label alignment might be approximate.", RuntimeWarning)
                    # Fallback: Use indices relative to the start of the *sampled* background data - less accurate if original df had gaps etc.
                    end_indices = [i + self.sequence_length - 1 for i in range(num_sequences)]


            valid_end_indices = [idx for idx in end_indices if idx < len(training_df_with_labels)]

            if len(valid_end_indices) != num_sequences:
                mismatch_warning = (
                    f"Potential mismatch after label indexing: Trying to get {num_sequences} labels, "
                    f"but only {len(valid_end_indices)} valid indices found (max index in training_df: {len(training_df_with_labels)-1}). "
                    f"Check sequence generation and sampling logic. Attempting to use available labels."
                )
                warnings.warn(mismatch_warning, RuntimeWarning)
                # Adjust background_data_np to match the number of labels we could find
                if len(valid_end_indices) < num_sequences:
                    print(f"Reducing background sequences from {num_sequences} to {len(valid_end_indices)} to match available labels.")
                    # This requires knowing *which* sequences correspond to the valid_end_indices.
                    # If indices were sampled, this is complex. Easiest safe approach:
                    # Re-slice background_data_np IF we know the corresponding rows.
                    # Safer fallback: If mismatch is large, maybe error out? For now, try extracting labels and see if TSExplainer handles it.
                    # Let's assume TSExplainer might handle slightly mismatched lengths or we proceed with fewer samples.
                    # We'll use the labels we could find.

            try:
                background_outcomes_np = training_df_with_labels[self.actual_label_col].iloc[valid_end_indices].values
                print(f"Extracted {len(background_outcomes_np)} background labels corresponding to sequences.")
                # If lengths still mismatch after extraction, trim the longer one
                if len(background_outcomes_np) != background_data_np.shape[0]:
                    min_len = min(len(background_outcomes_np), background_data_np.shape[0])
                    print(f"Adjusting background data/outcomes to matched length: {min_len}")
                    background_data_np = background_data_np[:min_len]
                    background_outcomes_np = background_outcomes_np[:min_len]

            except KeyError:
                raise KeyError(f"Label column '{self.actual_label_col}' not found in training_df_with_labels.")
            except IndexError as e:
                print(f"Detailed IndexError during label extraction: {e}")
                raise IndexError("Error accessing labels using calculated end indices. Check sequence alignment and sampling logic.")
            except Exception as e:
                print(f"Unexpected error during background label extraction: {e}")
                raise

            # --- 2. Initialize TimeSeriesExplainer ---
            try:
                print(f"Initializing TimeSeriesExplainer with mode '{self.mode}', SHAP method '{shap_method}'...")
                ts_explainer = TimeSeriesExplainer(
                    model=self.model_wrapper,
                    background_data=background_data_np,
                    background_outcomes=background_outcomes_np, # Pass the extracted labels
                    feature_names=self.feature_columns,
                    mode=self.mode,
                    # --- Pass DiCE specific context as kwargs ---
                    training_df_for_dice=training_df_with_labels, # Pass the df with labels
                    outcome_name_for_dice=self.actual_label_col, # Pass the label col name
                    continuous_features_for_dice=self.continuous_features_list,
                    # --- Shap Explainer Method ---
                    shap_method=shap_method
                )
                print("TimeSeriesExplainer initialized successfully.")
            except Exception as e:
                print(f"Failed to initialize TimeSeriesExplainer: {e}")
                import traceback
                traceback.print_exc()
                return # Stop if explainer fails

            # --- 3. Prepare Instances for Explanation ---
            print(f"Preparing instances for explanation from data source (shape: {data_source_for_explanation.shape}).")
            print(f"Using columns for sequence generation: {self.feature_columns}")

            # Ensure source data has required columns
            missing_cols = [col for col in self.feature_columns if col not in data_source_for_explanation.columns]
            if missing_cols:
                raise ValueError(f"Data source for explanation is missing required feature columns: {missing_cols}")

            # Generate ALL possible sequences from the source data ONCE
            all_instances_to_explain_np = ut.dataframe_to_sequences(
                df=data_source_for_explanation,
                sequence_length=self.sequence_length,
                feature_cols=self.feature_columns
            )

            num_instances_available = all_instances_to_explain_np.shape[0]
            if num_instances_available == 0:
                print("Warning: No sequences generated for explanation from the provided data source. Skipping explanation methods.")
                return # Skip if no instances

            print(f"Generated {num_instances_available} total sequences (instances) for potential explanation.")

            # --- Extract Corresponding True Labels for ALL potential Explained Instances ---
            # Calculate end indices in the source DataFrame for ALL sequences
            all_original_labels = None
            try:
                start_index_in_df = self.sequence_length - 1
                end_indices_for_explanation = list(range(start_index_in_df, start_index_in_df + num_instances_available))

                max_source_index = len(data_source_for_explanation) - 1
                valid_explain_indices = [idx for idx in end_indices_for_explanation if idx <= max_source_index]

                if not valid_explain_indices:
                    raise IndexError(f"No valid indices found for explanation labels (needed up to {end_indices_for_explanation[-1]}, max is {max_source_index}).")

                # Adjust available instances if not all labels could be found (should ideally not happen if sequence generation is correct)
                if len(valid_explain_indices) < num_instances_available:
                    warnings.warn(f"Could only find valid indices for {len(valid_explain_indices)} out of {num_instances_available} generated sequences. Trimming sequences.", RuntimeWarning)
                    num_instances_available = len(valid_explain_indices)
                    all_instances_to_explain_np = all_instances_to_explain_np[:num_instances_available]
                    end_indices_for_explanation = valid_explain_indices # Use the valid indices

                if self.actual_label_col not in data_source_for_explanation.columns:
                    raise KeyError(f"Label column '{self.actual_label_col}' not found in data_source_for_explanation DataFrame.")

                all_original_labels = data_source_for_explanation[self.actual_label_col].iloc[end_indices_for_explanation].values
                print(f"Successfully extracted {len(all_original_labels)} true labels corresponding to the {num_instances_available} generated sequences.")

            except (KeyError, IndexError, Exception) as label_err:
                print(f"ERROR extracting original labels for handler: {label_err}")
                print("Proceeding without original labels for the handler (some plots may be affected).")
                all_original_labels = None # Ensure it's None if extraction failed


            # --- 4. Define Plot Handlers Dictionary ---
            plot_handlers = {
                "ShapExplainer": x.process_and_plot_shap,
                "LimeExplainer": x.process_and_plot_lime,
                "DiceExplainer": x.process_and_plot_dice
            }

            # --- 5. Loop Through XAI Methods from Config ---
            for xai_config in self.xai_params:
                method_name = xai_config.get("method")
                settings = xai_config.get("settings", {}) # Use method-specific settings

                if not method_name or method_name == "none":
                    print("Skipping entry with no method name.")
                    continue

                print(f"\n===== Running Method: {method_name.upper()} =====")
                try:
                    # --- Per-Method Instance Limiting ---
                    n_explain_max_config = settings.get("n_explain_max") # Get value, could be None
                    # Provide a default if the key is missing OR if the value is None
                    if n_explain_max_config is None:
                        n_explain_max = 10 # Default number of instances if not specified or None
                        print(f"Using default n_explain_max={n_explain_max} for {method_name}.")
                    elif not isinstance(n_explain_max_config, int) or n_explain_max_config < 0:
                         warnings.warn(f"Invalid n_explain_max value '{n_explain_max_config}' for {method_name}. Using default=10.", RuntimeWarning)
                         n_explain_max = 10
                    else:
                        n_explain_max = n_explain_max_config

                    # Determine the actual number of instances for *this* method
                    num_instances_for_method = min(num_instances_available, n_explain_max)

                    if num_instances_for_method <= 0:
                         print(f"Skipping {method_name} as number of instances to explain is {num_instances_for_method}.")
                         continue # Skip to next method if no instances needed/available

                    print(f"Selected {num_instances_for_method} instance(s) for explanation with {method_name} (max requested: {n_explain_max}).")

                    # Slice the instances for this specific method
                    # Take the *first* N instances from the total generated pool
                    current_instances_np = all_instances_to_explain_np[:num_instances_for_method]

                    # Slice the corresponding labels if they exist
                    current_original_labels = None
                    if all_original_labels is not None:
                        current_original_labels = all_original_labels[:num_instances_for_method]
                        # Sanity check length
                        if len(current_original_labels) != current_instances_np.shape[0]:
                             warnings.warn(f"Label slice length mismatch for {method_name}. Check logic.", RuntimeWarning)
                             # Adjust to min length to avoid index errors downstream
                             min_len = min(len(current_original_labels), current_instances_np.shape[0])
                             current_instances_np = current_instances_np[:min_len]
                             current_original_labels = current_original_labels[:min_len]
                             num_instances_for_method = min_len # Update the count being processed

                    # --- End Per-Method Instance Limiting ---

                    # Get the specific explainer (SHAP, LIME, DiCE) from TimeSeriesExplainer
                    explainer_object = ts_explainer._get_or_initialize_explainer(method_name)
                    if explainer_object is None:
                        print(f"Could not get or initialize explainer for {method_name}. Skipping.")
                        continue

                    print(f"Using configuration for {method_name}: {settings}")

                    # Prepare common handler arguments
                    handler_args = {
                        "explainer_object": explainer_object,
                        "feature_names": self.feature_columns,
                        "sequence_length": self.sequence_length,
                        "output_dir": self.output_dir,
                        "mode": self.mode,
                        "job_name": self.job_name,
                        # Pass the correctly sliced labels for this method
                        "original_labels": current_original_labels
                    }

                    # --- Method Specific Logic ---
                    if method_name == "DiceExplainer":
                        features_to_vary = settings.get('features_to_vary', self.feature_columns) # Default to all features
                        dice_runtime_kwargs = {
                            'total_CFs': settings.get('total_CFs', 4),
                            'desired_class': settings.get('desired_class', 'opposite'),
                            'features_to_vary': features_to_vary,
                        }
                        print(f"DiCE Runtime Params: {dice_runtime_kwargs}")

                        start_explain_time = time.perf_counter()
                        # DiCE explains all instances provided in one go
                        xai_results = ts_explainer.explain(
                            instances_to_explain=current_instances_np, # Explain the selected N instances
                            method_name=method_name,
                            **dice_runtime_kwargs
                        )
                        end_explain_time = time.perf_counter()
                        print(f"DICE explanation took {end_explain_time - start_explain_time:.2f}s")

                        # Update handler args specific to DiCE results/instances
                        handler_args.update({
                            "results": xai_results,
                            "instances_explained": current_instances_np
                        })

                        # Call DiCE Handler
                        handler_func = plot_handlers.get(method_name)
                        if handler_func:
                            print("Calling DiCE plot handler...")
                            handler_func(**handler_args)
                        else:
                            print(f"No plot handler found for {method_name}")

                    elif method_name == "ShapExplainer":
                        # Determine nsamples and l1_reg based on potentially updated shap_method
                        current_shap_method = settings.get('shap_method', shap_method) # Allow override per config item
                        n_samples_shap = settings.get('nsamples', 50) # Use config or default
                        k_features = settings.get('l1_reg_k_features', 20)

                        # Adjust nsamples if using TreeSHAP - typically wants fewer or uses background size
                        if current_shap_method == 'tree':
                            # TreeSHAP doesn't use 'nsamples' in the same way, more about background data.
                            # Might need different parameters. Let's pass 'auto' for now.
                            n_samples_shap = 'auto' # Or adjust based on TreeExplainer specifics
                            l1_reg_shap = 'auto' # l1_reg less common/needed for TreeSHAP
                            print("Adjusting SHAP params for TreeExplainer.")
                        else: # KernelSHAP etc.
                            n_flat_features = self.sequence_length * len(self.feature_columns)
                            # Use feature count based l1_reg only if nsamples < n_flat_features
                            l1_reg_shap = f'num_features({k_features})' if n_samples_shap < n_flat_features else 'auto'

                        shap_runtime_kwargs = {
                            'nsamples': n_samples_shap,
                            'l1_reg': l1_reg_shap,
                        }
                        print(f"SHAP Runtime Params ({current_shap_method}): {shap_runtime_kwargs}")

                        start_explain_time = time.perf_counter()
                        # SHAP explains all selected instances in one batch
                        xai_results = ts_explainer.explain(
                            instances_to_explain=current_instances_np,
                            method_name=method_name,
                            **shap_runtime_kwargs
                        )
                        end_explain_time = time.perf_counter()
                        print(f"SHAP explanation ({current_shap_method}) took {end_explain_time - start_explain_time:.2f}s")

                        # Update handler args specific to SHAP
                        handler_args.update({
                            "results": xai_results, # SHAP values
                            "instances_explained": current_instances_np
                        })

                        handler_func = plot_handlers.get(method_name)
                        if handler_func:
                            print(f"Calling plot handler for SHAP...")
                            handler_func(**handler_args)
                        else:
                            print(f"No plot handler defined for SHAP.")

                    elif method_name == "LimeExplainer":
                        num_features = settings.get('num_features', 15) # K for LIME
                        num_samples = settings.get('num_samples', 1000) # Perturbations
                        lime_runtime_kwargs = {
                            'num_features': num_features,
                            'num_samples': num_samples
                        }
                        print(f"LIME Runtime Params: {lime_runtime_kwargs}")
                        print(f"Configuring LIME for {len(current_instances_np)} instance(s)...")


                        # INNER LOOP FOR LIME INSTANCES
                        for instance_idx in range(len(current_instances_np)):
                            print(f"--- Explaining Instance {instance_idx} with LIME ---")
                            # Select the single instance (needs to keep 3D shape: [1, seq_len, n_features])
                            current_instance_np = current_instances_np[instance_idx : instance_idx + 1]
                            if current_instance_np.size == 0:
                                print(f"Skipping LIME for instance {instance_idx} due to empty data.")
                                continue

                            # Extract the single corresponding original label if available
                            current_original_label = None
                            if current_original_labels is not None and instance_idx < len(current_original_labels):
                                current_original_label = [current_original_labels[instance_idx]] # Keep as list/array


                            try:
                                start_lime_time = time.perf_counter()
                                # Explain this single instance slice
                                lime_explanation_object = ts_explainer.explain(
                                    instances_to_explain=current_instance_np,
                                    method_name='LimeExplainer',
                                    **lime_runtime_kwargs
                                )
                                end_lime_time = time.perf_counter()
                                print(f"LIME explanation for instance {instance_idx} took {end_lime_time - start_lime_time:.2f}s")

                                # Update handler args for this specific LIME instance
                                handler_args.update({
                                    "results": lime_explanation_object, # LIME explanation object
                                    "instances_explained": current_instance_np, # The single instance explained
                                    "instance_index": instance_idx, # Pass index for unique file naming etc.
                                    "original_labels": current_original_label # Pass the single label
                                })


                                handler_func = plot_handlers.get(method_name)
                                if handler_func:
                                    print(f"Calling plot handler for LIME instance {instance_idx}...")
                                    # Pass instance_index to handler if needed
                                    handler_func(**handler_args)
                                else:
                                    print(f"No plot handler defined for LIME.")

                            except Exception as lime_instance_err:
                                print(f"ERROR during LIME explanation/plotting for instance {instance_idx}: {lime_instance_err}")
                                import traceback
                                traceback.print_exc() # Print traceback for debugging

                        print(f"--- Finished LIME Explanations ---")
                        # END INNER LOOP FOR LIME

                    else:
                        print(f"Skipping unknown or unhandled XAI method: {method_name}")

                except Exception as explain_err:
                    print(f"ERROR during explanation/plotting setup for method '{method_name}': {explain_err}")
                    import traceback
                    traceback.print_exc()

            # End loop through xai_params configs

        except Exception as xai_general_err:
            print(f"ERROR during XAI processing in XAIRunner: {xai_general_err}")
            import traceback
            traceback.print_exc()

        print(f"--- Finished XAI Execution via XAIRunner for job '{self.job_name}' ---")