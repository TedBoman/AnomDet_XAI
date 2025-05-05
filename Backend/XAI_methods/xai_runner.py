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
MAX_BG_SAMPLES = 250000 # Example default, can be overridden

class XAIRunner:
    """
    Handles the execution of Explainable AI (XAI) methods for time series models.
    """
    def __init__(
        self,
        xai_settings: Dict[str, Any],
        model_wrapper: ModelWrapperForXAI,
        sequence_length: int,
        feature_columns: List[str],
        actual_label_col: str,
        continuous_features_list: List[str],
        job_name: str,
        mode: str = 'classification', # Default mode
        output_dir: str = OUTPUT_DIR,
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
        if not isinstance(xai_settings, dict):
            raise TypeError("xai_settings must be a dictionary wit settings and list of dictionaries.")
        if not model_wrapper:
            raise ValueError("model_wrapper cannot be None.")
        if not isinstance(sequence_length, int) or sequence_length <= 0:
            # Even non-sequential models need a sequence context for XAI framework
            raise ValueError("sequence_length must be a positive integer for XAI processing.")

        self.xai_params = xai_settings.get("xai_params", None)
        self.model_wrapper = model_wrapper
        self.sequence_length = sequence_length
        self.feature_columns = feature_columns
        self.actual_label_col = actual_label_col
        self.continuous_features_list = continuous_features_list
        self.job_name = job_name
        self.mode = mode
        self.output_dir = output_dir
        
        self.xai_settings = xai_settings.get("xai_settings")
        self.xai_num_samples = xai_settings.get("xai_num_samples", 10)
        self.xai_sampling_strategy = xai_settings.get("xai_sampling_strategy", "random")

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
                    shap_method = settings.get('shap_method', 'kernel')

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

            # --- 3. Prepare ALL Instances and Labels from Explanation Source ---
            all_instances_to_explain_np = np.array([])
            all_original_labels = None
            num_instances_available = 0
            try:
                print(f"Generating all possible sequences from explanation data source...")
                all_instances_to_explain_np = ut.dataframe_to_sequences(df=data_source_for_explanation, sequence_length=self.sequence_length, feature_cols=self.feature_columns)
                num_instances_available = all_instances_to_explain_np.shape[0]
                if num_instances_available == 0: print("Warning: No sequences generated for explanation."); return
                print(f"Generated {num_instances_available} total sequences for potential explanation.")

                # Extract Labels corresponding to the END of each sequence
                end_indices_exp = list(range(self.sequence_length - 1, self.sequence_length - 1 + num_instances_available))
                max_source_index_exp = len(data_source_for_explanation) - 1
                valid_end_indices_exp = [idx for idx in end_indices_exp if idx <= max_source_index_exp]

                if len(valid_end_indices_exp) < num_instances_available:
                    warnings.warn(f"Label index mismatch for explanation source ({len(valid_end_indices_exp)} vs {num_instances_available}). Trimming sequences.", RuntimeWarning)
                    num_instances_available = len(valid_end_indices_exp)
                    all_instances_to_explain_np = all_instances_to_explain_np[:num_instances_available]

                if self.actual_label_col in data_source_for_explanation.columns:
                    all_original_labels = data_source_for_explanation[self.actual_label_col].iloc[valid_end_indices_exp].values
                    print(f"Extracted {len(all_original_labels)} labels for the {num_instances_available} generated sequences.")
                    if len(all_original_labels) != all_instances_to_explain_np.shape[0]: # Final length check
                        min_len = min(len(all_original_labels), all_instances_to_explain_np.shape[0])
                        warnings.warn(f"Final label/instance mismatch. Adjusting to {min_len}.", RuntimeWarning)
                        all_instances_to_explain_np = all_instances_to_explain_np[:min_len]
                        all_original_labels = all_original_labels[:min_len]; num_instances_available = min_len
                else: warnings.warn(f"Label column '{self.actual_label_col}' not found in explanation source.")

            except Exception as prep_err: print(f"ERROR preparing instances/labels from source: {prep_err}"); traceback.print_exc(); return

            # --- 4. Define Plot Handlers Dictionary ---
            plot_handlers = {
                "ShapExplainer": x.process_and_plot_shap,
                "LimeExplainer": x.process_and_plot_lime,
                "DiceExplainer": x.process_and_plot_dice
            }

            # --- 5. Loop Through XAI Methods from Config ---
            for xai_config in self.xai_params:
                method_name = xai_config.get("method")
                settings = xai_config.get("settings", {})

                if not method_name or method_name == "none": continue
                print(f"\n===== Running Method: {method_name.upper()} =====")

                try:
                    # --- *** Determine Indices TO Explain for THIS method *** ---
                    final_sequence_indices = np.array([], dtype=int)
                    method_specific_orig_indices = settings.get('explain_indices') # Check for override list

                    if method_specific_orig_indices is not None and isinstance(method_specific_orig_indices, list):
                        print(f"Using method-specific indices from config: {method_specific_orig_indices[:10]}...")
                        # Ensure indices are integers and valid for the source DF
                        selected_original_indices = np.array([int(i) for i in method_specific_orig_indices if i in data_source_for_explanation.index])
                        if len(selected_original_indices) != len(method_specific_orig_indices):
                            warnings.warn("Some method-specific indices were out of bounds for the source DataFrame.", RuntimeWarning)
                    else:
                        # Use global strategy defined in __init__
                        current_strategy = settings.get('sampling_strategy', self.xai_sampling_strategy) # Allow override of strategy per method? No, use global.
                        current_n_samples = settings.get('num_samples', self.xai_num_samples) # Allow override of N per method? No, use global N.
                        print(f"Using global sampling strategy '{self.xai_sampling_strategy}' with n={self.xai_num_samples}.")
                        selected_original_indices = ut.select_explanation_indices(
                            data_source_for_explanation, # Sample from the full source
                            self.xai_sampling_strategy,
                            self.xai_num_samples,
                            label_col=self.actual_label_col
                        )

                    if len(selected_original_indices) == 0:
                        print(f"WARNING: No original indices selected based on strategy/override for {method_name}. Skipping method.")
                        continue

                    # --- Map selected original DF indices to sequence array indices ---
                    # Sequence 'j' corresponds to label at original index 'j + sequence_length - 1'
                    # So, to find sequence index 'j' for original index 'idx', use j = idx - (sequence_length - 1)
                    offset = self.sequence_length - 1
                    sequence_indices_potential = selected_original_indices - offset
                    # Filter valid sequence indices (must be >= 0 and < num_sequences_available)
                    valid_mask = (sequence_indices_potential >= 0) & (sequence_indices_potential < num_instances_available)
                    final_sequence_indices = sequence_indices_potential[valid_mask].astype(int)

                    if len(final_sequence_indices) < len(selected_original_indices):
                        warnings.warn(f"Could not map all selected original indices to valid sequence indices (mapped {len(final_sequence_indices)} / {len(selected_original_indices)}). Some instances might be too close to the start.", RuntimeWarning)

                    if len(final_sequence_indices) == 0:
                        print(f"WARNING: No valid sequence indices derived from selection for {method_name}. Skipping method.")
                        continue

                    # --- Apply per-method instance limit (n_explain_max) ---
                    # This limit applies *after* the sampling strategy/override
                    n_explain_max = settings.get("n_explain_max")
                    if n_explain_max is not None:
                        try:
                            n_explain_max = int(n_explain_max)
                            if n_explain_max > 0 and n_explain_max < len(final_sequence_indices):
                                print(f"Applying method-specific limit n_explain_max={n_explain_max} (selected {len(final_sequence_indices)} initially). Taking first N.")
                                # Take first N of the selected/mapped sequence indices
                                final_sequence_indices = final_sequence_indices[:n_explain_max]
                            elif n_explain_max <= 0:
                                warnings.warn(f"Invalid n_explain_max value {n_explain_max}. Ignoring limit.", RuntimeWarning)
                        except (ValueError, TypeError):
                            warnings.warn(f"Invalid n_explain_max value '{n_explain_max}'. Ignoring limit.", RuntimeWarning)

                    num_sequences_to_process = len(final_sequence_indices)
                    if num_sequences_to_process == 0:
                        print(f"No instances left to explain for {method_name} after limiting/mapping. Skipping.")
                        continue

                    print(f"Final number of sequences to explain for {method_name}: {num_sequences_to_process}")
                    # print(f"Final sequence indices: {final_sequence_indices[:10]}...") # Debug

                    # --- Slice the data and labels for the current method ---
                    current_instances_np = all_instances_to_explain_np[final_sequence_indices]
                    current_original_labels = None
                    if all_original_labels is not None:
                        try:
                            current_original_labels = all_original_labels[final_sequence_indices]
                            if len(current_original_labels) != len(current_instances_np): # Sanity check
                                warnings.warn("Final label/instance slice mismatch after n_explain_max. Check index logic.", RuntimeWarning)
                                min_len_final = min(len(current_original_labels), len(current_instances_np))
                                current_instances_np = current_instances_np[:min_len_final]
                                current_original_labels = current_original_labels[:min_len_final]
                                num_sequences_to_process = min_len_final
                        except IndexError as slice_err:
                            print(f"ERROR slicing labels with final sequence indices: {slice_err}. Proceeding without labels for this method.")
                            current_original_labels = None # Ensure it's None if slicing fails

                    # --- *** END Index Selection / Data Slicing *** ---

                    # Get the specific explainer from TimeSeriesExplainer
                    explainer_object = ts_explainer._get_or_initialize_explainer(method_name)
                    if explainer_object is None: print(f"Could not get explainer for {method_name}. Skipping."); continue

                    print(f"Using configuration for {method_name}: {settings}")
                    handler_args = { # Prepare common args for plot handler
                        "explainer_object": explainer_object, "feature_names": self.feature_columns,
                        "sequence_length": self.sequence_length, "output_dir": self.output_dir,
                        "mode": self.mode, "job_name": self.job_name,
                    }

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
                        # (DiCE logic using current_instances_np)
                        features_to_vary = settings.get('features_to_vary', self.feature_columns)
                        dice_runtime_kwargs = {'total_CFs': settings.get('total_CFs', 4), 'desired_class': settings.get('desired_class', 'opposite'), 'features_to_vary': features_to_vary}
                        print(f"DiCE Runtime Params: {dice_runtime_kwargs}")
                        start_explain_time = time.perf_counter()
                        xai_results = ts_explainer.explain(instances_to_explain=current_instances_np, method_name=method_name, **dice_runtime_kwargs)
                        end_explain_time = time.perf_counter(); print(f"DICE explanation took {end_explain_time - start_explain_time:.2f}s")
                        handler_args.update({"results": xai_results, "instances_explained": current_instances_np, "original_labels": current_original_labels})
                        handler_func = plot_handlers.get(method_name);
                        if handler_func: print("Calling DiCE plot handler..."); handler_func(**handler_args)
                        else: print(f"No plot handler found for {method_name}")

                    elif method_name == "ShapExplainer":
                        # (SHAP logic using current_instances_np)
                        current_shap_method = settings.get('shap_method', shap_method)
                        n_samples_shap = settings.get('nsamples', 50); k_features = settings.get('l1_reg_k_features', 20)
                        if current_shap_method == 'tree': n_samples_shap = 'auto'; l1_reg_shap = 'auto'
                        else: n_flat_features = self.sequence_length * len(self.feature_columns); l1_reg_shap = f'num_features({k_features})' if n_samples_shap < n_flat_features else 'auto'
                        shap_runtime_kwargs = {'nsamples': n_samples_shap, 'l1_reg': l1_reg_shap}
                        print(f"SHAP Runtime Params ({current_shap_method}): {shap_runtime_kwargs}")
                        start_explain_time = time.perf_counter()
                        xai_results = ts_explainer.explain(instances_to_explain=current_instances_np, method_name=method_name, **shap_runtime_kwargs)
                        end_explain_time = time.perf_counter(); print(f"SHAP explanation ({current_shap_method}) took {end_explain_time - start_explain_time:.2f}s")
                        handler_args.update({"results": xai_results, "instances_explained": current_instances_np, "original_labels": current_original_labels})
                        handler_func = plot_handlers.get(method_name);
                        if handler_func: print(f"Calling plot handler for SHAP..."); handler_func(**handler_args)
                        else: print(f"No plot handler defined for SHAP.")

                    elif method_name == "LimeExplainer":
                        # (LIME logic looping through current_instances_np)
                        num_features = settings.get('num_features', 15); num_samples = settings.get('num_samples', 1000)
                        lime_runtime_kwargs = {'num_features': num_features, 'num_samples': num_samples}
                        print(f"LIME Runtime Params: {lime_runtime_kwargs}")
                        print(f"Executing LIME for {num_sequences_to_process} selected instance(s)...")

                        for loop_idx in range(num_sequences_to_process):
                            # Get the original sequence index for context if needed
                            original_sequence_idx = final_sequence_indices[loop_idx]
                            print(f"--- Explaining Instance LoopIdx={loop_idx} (OrigSeqIdx={original_sequence_idx}) with LIME ---")
                            current_instance_np_single = current_instances_np[loop_idx : loop_idx + 1]
                            if current_instance_np_single.size == 0: continue

                            current_original_label_single = None
                            if current_original_labels is not None:
                                current_original_label_single = [current_original_labels[loop_idx]] # Keep as list

                            try:
                                start_lime_time = time.perf_counter()
                                lime_explanation_object = ts_explainer.explain(instances_to_explain=current_instance_np_single, method_name='LimeExplainer', **lime_runtime_kwargs)
                                end_lime_time = time.perf_counter(); print(f"LIME explanation for instance {loop_idx} took {end_lime_time - start_lime_time:.2f}s")

                                handler_args.update({
                                    "results": lime_explanation_object, "instances_explained": current_instance_np_single,
                                    "instance_index": loop_idx, # Use loop index for file naming
                                    "original_labels": current_original_label_single
                                })
                                handler_func = plot_handlers.get(method_name);
                                if handler_func: print(f"Calling plot handler for LIME instance {loop_idx}..."); handler_func(**handler_args)
                                else: print(f"No plot handler defined for LIME.")
                            except Exception as lime_instance_err:
                                print(f"ERROR during LIME explanation/plotting for instance {loop_idx} (OrigSeqIdx: {original_sequence_idx}): {lime_instance_err}")
                                traceback.print_exc()
                        print(f"--- Finished LIME Explanations ---")

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