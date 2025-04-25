# File: xai_visualizations.py

import traceback
import shap
import dice_ml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
from typing import Any, List, Optional

# Assuming ExplainerMethodAPI is defined elsewhere if needed for type hints on explainer_object
# from explainer_method_api import ExplainerMethodAPI

# --- SHAP Handler ---
def process_and_plot_shap(
    results: Any,                       # Expect np.ndarray or list[np.ndarray] from ShapExplainer.explain
    explainer_object: Any,              # The specific ShapExplainer instance (to get expected_value)
    instances_explained: np.ndarray,    # 3D numpy array (n, seq, feat) that was explained
    feature_names: List[str],           # Base feature names (e.g., ['Temp', 'Pressure'])
    sequence_length: int,
    output_dir: str,
    mode: str,                          # 'classification' or 'regression'
    class_index_to_plot: int = 0,       # Default class to plot for classification
    max_display_features: int = 20,     
    # Max features for plots like bar/waterfall
    job_name='none',
    ):
    """Processes SHAP results (already reshaped) and generates standard plots."""
    print(f"--- Processing and Plotting SHAP Results (Class Index: {class_index_to_plot if mode=='classification' else 'N/A'}) ---")

    if results is None:
        print("SHAP results are None. Skipping plotting.")
        return
    
    output_dir = output_dir+'/SHAP'

    # --- Prepare Data for Standard SHAP Plots ---
    n_instances_explained = instances_explained.shape[0]
    n_features = len(feature_names)
    n_flat_features = sequence_length * n_features

    feature_names_flat = [f"{feat}_t-{i}" for feat in feature_names for i in range(sequence_length -1, -1, -1)]
    features_flat_np = instances_explained.reshape(n_instances_explained, -1)

    is_classification = isinstance(results, list)
    shap_values_3d = None
    expected_value_for_plot = None
    base_expected_value = getattr(explainer_object, 'expected_value', None) # Use the property

    if is_classification:
        if not results or class_index_to_plot >= len(results):
            print(f"SHAP results list is empty or invalid class index {class_index_to_plot}. Skipping plotting.")
            return
        shap_values_3d = results[class_index_to_plot]
        if base_expected_value is not None and hasattr(base_expected_value, '__len__') and len(base_expected_value) > class_index_to_plot:
            expected_value_for_plot = base_expected_value[class_index_to_plot]
        else: warnings.warn(f"Could not get expected value for class {class_index_to_plot}.", RuntimeWarning)
    else: # Regression or Binary assumed to return single array
        shap_values_3d = results
        expected_value_for_plot = base_expected_value

    if shap_values_3d is None or shap_values_3d.size == 0:
         print("No valid SHAP values found for the selected class/output. Skipping plotting.")
         return

    # --- Final Validation and Reshaping for Plotting ---
    expected_shape_3d = (n_instances_explained, sequence_length, n_features)
    if shap_values_3d.shape != expected_shape_3d:
         print(f"Warning: SHAP values shape {shap_values_3d.shape} mismatch expected {expected_shape_3d}. Check explainer's output.")
         try: # Attempt reshape if possible
             shap_values_flat = shap_values_3d.reshape(n_instances_explained, -1)
             if shap_values_flat.shape[1] != n_flat_features: raise ValueError("Flattened shape mismatch.")
         except ValueError as e:
             print(f"Cannot proceed with plotting due to shape mismatch: {e}")
             return
    else:
         shap_values_flat = shap_values_3d.reshape(n_instances_explained, -1)


    # --- Generate and Save Plots ---
    print(f"Saving SHAP plots to {output_dir}...")
    plot_suffix = f"_c{class_index_to_plot}" if is_classification else ""
    os.makedirs(output_dir, exist_ok=True)

    # Create SHAP Explanation object (handles missing base values better)
    shap_explanation = None
    try:
         shap_explanation = shap.Explanation(
             values=shap_values_flat,
             base_values=expected_value_for_plot, # Pass base value if available
             data=features_flat_np,
             feature_names=feature_names_flat
         )
    except Exception as e:
         print(f"Could not create shap.Explanation object: {e}. Some plots may require it.")

    # Plotting functions (summary_plot, bar, waterfall, heatmap)
    # Wrap each plot in try/except and ensure plt.close()
    # --- Summary Plot (Dot) ---
    try:
        plt.figure()
        shap.summary_plot(shap_values_flat, features=features_flat_np, feature_names=feature_names_flat, show=False, plot_type='dot')
        plt.title(f"SHAP Summary Plot (Dot{plot_suffix})")
        plt.savefig(os.path.join(output_dir, f"{job_name}_shap_summary_dot{plot_suffix}.png"), bbox_inches='tight')
        print("Saved Summary Plot (Dot).")
    except Exception as e: print(f"Failed Summary Plot (Dot): {e}")
    finally: plt.close() # Ensure figure is closed

    # --- Bar Plot ---
    if shap_explanation:
        try:
            plt.figure()
            shap.plots.bar(shap_explanation, max_display=max_display_features, show=False)
            plt.title(f"SHAP Feature Importance (Bar{plot_suffix})")
            plt.savefig(os.path.join(output_dir, f"{job_name}_shap_summary_bar{plot_suffix}.png"), bbox_inches='tight')
            print("Saved Summary Plot (Bar).")
        except Exception as e: print(f"Failed Summary Plot (Bar): {e}")
        finally: plt.close()
    else: print("Skipping Bar plot: shap.Explanation object needed.")

    # --- Waterfall Plot (First Instance) ---
    if shap_explanation and expected_value_for_plot is not None:
        instance_idx_to_plot = 0
        try:
            plt.figure()
            shap.plots.waterfall(shap_explanation[instance_idx_to_plot], max_display=max_display_features, show=False)
            plt.savefig(os.path.join(output_dir, f"{job_name}_shap_waterfall_inst{instance_idx_to_plot}{plot_suffix}.png"), bbox_inches='tight')
            print(f"Saved Waterfall Plot for Instance {instance_idx_to_plot}.")
        except Exception as e: print(f"Failed Waterfall Plot: {e}")
        finally: plt.close()
    elif expected_value_for_plot is None: print("Skipping Waterfall plot: expected_value not available.")
    else: print("Skipping Waterfall plot: shap.Explanation object needed.")


    # --- Heatmap Plot ---
    if shap_explanation:
        try:
            plt.figure()
            # Heatmap often needs more vertical space
            fig_height = max(6, n_instances_explained * 0.4) # Adjust height dynamically
            plt.figure(figsize=(8, fig_height)) # Set figure size before plotting
            shap.plots.heatmap(shap_explanation, max_display=min(max_display_features, n_flat_features), show=False)
            plt.savefig(os.path.join(output_dir, f"{job_name}_shap_heatmap{plot_suffix}.png"), bbox_inches='tight')
            print("Saved Heatmap Plot.")
        except Exception as e: print(f"Failed Heatmap Plot: {e}")
        finally: plt.close()
    else: print("Skipping Heatmap plot: shap.Explanation object needed.")

    print("--- Finished SHAP Plotting ---")


# --- LIME Handler ---
def process_and_plot_lime(
     results: Any,                       # Expect LIME Explanation object
     explainer_object: Any,              # The specific LimeExplainer instance
     instances_explained: np.ndarray,    # Should be shape (1, seq, feat) for LIME
     feature_names: List[str],           # Base feature names
     sequence_length: int,
     output_dir: str,
     mode: str,
     instance_index: int = 0,            # Index if looping outside
     job_name='none',
     **kwargs):
     """Processes LIME results and generates standard plots/output."""
     print(f"--- Processing and Plotting LIME Results for Instance Index {instance_index} ---")

     if results is None:
         print("LIME results object is None. Skipping.")
         return
     
     output_dir = output_dir+'/LIME'

     # LIME Explanation object usually comes from explaining one instance
     lime_explanation = results

     try:
         # Save as HTML file (most common way to save LIME plots)
         os.makedirs(output_dir, exist_ok=True) # Ensure dir exists
         html_file = os.path.join(output_dir, f"{job_name}_lime_explanation_inst{instance_index}.html")
         lime_explanation.save_to_file(html_file)
         print(f"Saved LIME explanation HTML to {html_file}")

         # Print top features to console
         print(f"Top LIME features for instance {instance_index}:")
         print(lime_explanation.as_list())

     except Exception as e:
         print(f"Failed to process/save LIME results for instance {instance_index}: {e}")

     print("--- Finished LIME Plotting ---")

def process_and_plot_dice(
    results: Any, # Expect dice_ml.explanation.Explanation or similar structure
    explainer_object: Any, # The DiceExplainerWrapper instance (MUST have flat_feature_names, outcome_name)
    instances_explained: np.ndarray, # 3D numpy array (instance_idx, seq_len, features)
    feature_names: List[str], # Base feature names (used for dimension check)
    sequence_length: int, # Used for dimension check
    output_dir: str,
    mode: str, # Should be classification
    job_name: str = "job",
    # original_df: pd.Series=None, # Removed - we get original from instances_explained
    **kwargs):
    """
    Processes DiCE Explanation object. Saves a CSV for each instance containing
    the original instance as the first row and counterfactuals as subsequent rows.
    Adds a 'type' column for identification.
    """
    print(f"--- Processing and Saving DiCE Counterfactuals with Original ---")

    output_dir = output_dir+'/DiCE'

    os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists

    try:
        # --- Access CF results list ---
        if hasattr(results, 'cf_examples_list'):
             cf_examples_list = results.cf_examples_list
        elif isinstance(results, list):
             cf_examples_list = results
        elif hasattr(results, 'final_cfs_df'): # Handle single instance result
             cf_examples_list = [results]
        else:
             print("Error: Could not find counterfactual list/data in 'results' object.")
             return

        if not cf_examples_list:
             print("No counterfactual examples found in the results.")
             return

        print(f"Found explanation results for {len(cf_examples_list)} instance(s).")

        # --- Loop through each instance ---
        for i, cf_example in enumerate(cf_examples_list):
             print(f"  Processing instance {i}...")

             # --- Get Counterfactuals DataFrame ---
             cfs_df = None
             if hasattr(cf_example, 'final_cfs_df') and cf_example.final_cfs_df is not None:
                  cfs_df = cf_example.final_cfs_df.copy() # Use copy to avoid modifying original
             # Add elif for sparse if needed

             if cfs_df is None or cfs_df.empty:
                  print(f"    No counterfactuals DataFrame found or empty for instance {i}. Skipping CSV save.")
                  continue
             if not isinstance(cfs_df, pd.DataFrame):
                  print(f"    Warning: Expected pandas DataFrame for counterfactuals, got {type(cfs_df)}. Skipping CSV save for instance {i}.")
                  continue

             # Add 'type' column to counterfactuals
             cfs_df['type'] = 'counterfactual'
             flat_feature_names = cfs_df.columns.values

             # --- Prepare Original Instance Row ---
             try:
                  original_instance_3d = instances_explained[i]
                  # Flatten the original instance
                  original_flat_np = original_instance_3d.flatten()

                  # Create a pandas Series for the original instance
                  # Ensure index matches the flattened feature names from the explainer
                  original_series = pd.Series(original_flat_np, index=flat_feature_names)

                  # Add the type identifier
                  original_series['type'] = 'original'

                  # Convert Series to a DataFrame row, ensure columns match CFs
                  original_df_row = original_series.to_frame().T
                  # Reorder columns to match cfs_df potentially, ensure 'type' is present
                  original_df_row = original_df_row.astype(cfs_df.dtypes) # Match types


             except IndexError:
                  print(f"    Error: Cannot get original instance for index {i} from instances_explained (shape {instances_explained.shape}).")
                  continue
             except Exception as e_orig:
                  print(f"    Error processing original instance {i}: {e_orig}")
                  continue

             # --- Combine Original and Counterfactuals ---
             # Make 'type' the first column for clarity
             cols_cf = ['type'] + [col for col in cfs_df.columns if col != 'type']
             cfs_df = cfs_df[cols_cf]

             # Ensure original row has same columns in same order before concatenating
             original_df_row = original_df_row[cols_cf] # Match column order

             combined_df = pd.concat([original_df_row, cfs_df], ignore_index=True)

             # --- Save Combined CSV ---
             filename = f"{job_name}_instance_{i}_original_and_counterfactuals.csv"
             filepath = os.path.join(output_dir, filename)
             combined_df.to_csv(filepath, index=False)
             print(f"    Saved original and counterfactuals for instance {i} to: {filepath}")

    except AttributeError as e:
         print(f"Failed accessing attributes: {e}")
         traceback.print_exc()
    except Exception as e:
         print(f"Failed processing/saving results: {e}")
         traceback.print_exc()

    print("--- Finished DiCE Counterfactual Saving (with Original) ---")

# --- Add handlers for other XAI methods as needed ---