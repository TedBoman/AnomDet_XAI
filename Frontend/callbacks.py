import sys
import traceback
from dash import Dash, dcc, html, Input, Output, State, ALL, MATCH, callback, callback_context, no_update
import json
from get_handler import get_handler
import os
from ml_model_hyperparameters import HYPERPARAMETER_DESCRIPTIONS

def get_index_callbacks(app):

# --- NEW Callback to update Parameter Explanation Box ---
    @app.callback(
        Output("parameter-explanation-box", "children"),
        Input("detection-model-dropdown", "value")
    )
    def update_parameter_explanations(selected_model):
        if not selected_model or selected_model == 'none':
            return [html.P("Select a model to see parameter explanations.", style={'color':'#b0b0b0'})]

        descriptions = HYPERPARAMETER_DESCRIPTIONS.get(selected_model, {})
        if not descriptions:
            return [html.P(f"No descriptions available for model: {selected_model}", style={'color':'#ffcc00'})]

        explanation_children = [
            html.H5(f"{selected_model} Parameters:", style={'color':'#ffffff', 'marginBottom':'15px', 'borderBottom': '1px solid #555', 'paddingBottom':'5px'})
        ]

        if not descriptions:
             explanation_children.append(html.P("No descriptions found for this model.", style={'fontStyle': 'italic', 'color':'#aaaaaa'}))
        else:
            for param, desc in descriptions.items():
                explanation_children.append(
                    html.Div([
                        html.Strong(f"{param}:", style={'color':'#ffffff'}),
                        html.P(desc, style={'color':'#d0d0d0', 'marginTop':'2px', 'marginBottom':'10px', 'fontSize':'14px'})
                    ])
                )

        return explanation_children

    # --- Callback to toggle visibility of Injection panel ---
    @app.callback(
        Output("injection-panel", "style"), # Target the inner panel
        Input("injection-check", "value")
    )
    def toggle_injection_panel(selected_injection):
        if "use_injection" in selected_injection:
            # Return style to make it visible, keep other styles if needed
            return {"display": "block", "marginTop": "15px", "padding": "10px", "border": "1px solid #444", "borderRadius": "5px", "backgroundColor": "#145E88"}
        return {"display": "none"}

    # --- Callback to toggle visibility of Labeled panel ---
    @app.callback(
        Output("label-column-selection-div", "style"),
        Input("labeled-check", "value")
    )
    def toggle_labeled_panel(selected_labeled):
        if "is_labeled" in selected_labeled:
            return {"display": "block", "marginTop": "10px", "textAlign": "center"}
        return {"display": "none"}
    
    # --- Callback to generate dynamic ML model settings panel ---
    @app.callback(
        Output("model-settings-panel", "children"),
        Output("model-settings-panel", "style"), # Add style output to show/hide
        Input("detection-model-dropdown", "value")
    )
    def update_model_settings_panel(selected_model):
        if not selected_model:
            return [], {"display": "none"} # Hide if no model selected

        settings_children = [html.H5(f"Settings for {selected_model}:", style={'color':'#ffffff', 'marginBottom': '15px'})]
        panel_style = {"marginTop": "15px", "padding": "15px", "border": "1px solid #444", "borderRadius": "5px", "backgroundColor": "#145E88", "display": "block"} # Style to show panel

        # Use Pattern-Matching IDs for settings
        setting_id_base = {'type': 'ml-setting', 'model': selected_model}

        # Define settings based on selected_model
        if selected_model == "XGBoost":
                # Based on XGBoostModel in XGBoost.py which uses xgb.XGBClassifier
            settings_children.extend([
                # --- Core Booster Params ---
                html.Div([
                    html.Label("Num Estimators (n_estimators):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "210px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'n_estimators'},
                                type="number", value=100, min=10, step=10, # Default 100
                                style={'width':'150px', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                html.Div([
                    html.Label("Learning Rate (learning_rate):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "210px"}),
                        dcc.Input(id={**setting_id_base, 'param': 'learning_rate'},
                                  type="number",
                                  value=0.1,
                                  min=0.0,  # Changed min to 0.0
                                  step=0.01,
                                  style={'width':'150px', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                html.Div([
                    html.Label("Max Depth (max_depth):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "210px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'max_depth'},
                                type="number", value=6, min=1, step=1, # Default 6
                                style={'width':'150px', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                html.Div([
                    html.Label("Min Child Weight (min_child_weight):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "210px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'min_child_weight'},
                                type="number", value=1, min=0, step=1, # Default 1
                                style={'width':'150px', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                html.Div([
                    html.Label("Gamma (min_split_loss):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "210px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'gamma'},
                                type="number", value=0, min=0, step=0.1, # Default 0
                                style={'width':'150px', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # --- Sampling Params ---
                    html.Div([
                    html.Label("Subsample Ratio (subsample):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "210px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'subsample'},
                                type="number", value=1.0, min=0.1, max=1.0, step=0.05, # Default 1.0 
                                style={'width':'150px', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                    html.Div([
                    html.Label("Col Sample by Tree (colsample_bytree):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "210px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'colsample_bytree'},
                                type="number", value=1.0, min=0.1, max=1.0, step=0.05, # Default 1.0 - 
                                style={'width':'150px', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                    html.Div([
                    html.Label("Col Sample by Level (colsample_bylevel):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "210px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'colsample_bylevel'},
                                type="number", value=1.0, min=0.1, max=1.0, step=0.05, # Default 1.0
                                style={'width':'150px', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                    html.Div([
                    html.Label("Col Sample by Node (colsample_bynode):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "210px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'colsample_bynode'},
                                type="number", value=1.0, min=0.1, max=1.0, step=0.05, # Default 1.0
                                style={'width':'150px', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # --- Regularization ---
                    html.Div([
                    html.Label("L1 Reg Alpha (reg_alpha):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "210px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'reg_alpha'},
                                type="number", value=0, min=0, step=0.1, # Default 0
                                style={'width':'150px', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                    html.Div([
                    html.Label("L2 Reg Lambda (reg_lambda):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "210px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'reg_lambda'},
                                type="number", value=1, min=0, step=0.1, # Default 1
                                style={'width':'150px', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # --- Other Params ---
                html.Div([
                    html.Label("Booster Type:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "210px"}),
                    dcc.Dropdown(id={**setting_id_base, 'param': 'booster'},
                                    options=[{'label': b, 'value': b} for b in ['gbtree', 'gblinear', 'dart']],
                                    value='gbtree', clearable=False, # Default gbtree
                                    style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                html.Div([
                    html.Label("Random State (int):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "210px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'random_state'},
                                type="number", placeholder="None", step=1, # Allow None (empty) or int
                                style={'width':'150px', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                html.Div([
                    html.Label("Probability Calibration Method:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "210px"}),
                    dcc.Dropdown(id={**setting_id_base, 'param': 'calibration_method'},
                                    options=[{'label': m, 'value': m} for m in ['isotonic', 'sigmoid']],
                                    value='isotonic', clearable=False, # Default isotonic from backend
                                    style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                # Note: scale_pos_weight is handled automatically in the backend XGBoostModel run method based on labels
                # Note: objective and eval_metric are fixed in the backend XGBoostModel __init__ for now ('binary:logistic', 'logloss')
            ])
        elif selected_model == "lstm":
            # Based on LSTMModel in lstm.py
            settings_children.extend([
                # --- Architecture ---
                html.Div([
                    html.Label("LSTM Units per Layer (int):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'units'},
                                type="number", value=64, min=8, step=8, # Default from backend
                                style={'width':'150px', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                html.Div([
                    html.Label("LSTM Activation:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                    dcc.Dropdown(id={**setting_id_base, 'param': 'activation'},
                                    options=[{'label': act, 'value': act} for act in ['relu', 'tanh', 'sigmoid', 'elu']],
                                    value='relu', clearable=False, # Default from backend
                                    style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                html.Div([
                        html.Label("Dropout Rate (float 0-1):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                        dcc.Input(id={**setting_id_base, 'param': 'dropout'},
                                type="number", value=0.0, min=0.0, max=1.0, step=0.05, # Default 0.0 as backend doesn't use it yet
                                style={'width':'150px', 'verticalAlign':'middle'})
                    ], style={'marginBottom':'8px', 'textAlign':'left'}),
                    html.Div([
                        html.Label("Recurrent Dropout (float 0-1):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                        dcc.Input(id={**setting_id_base, 'param': 'recurrent_dropout'},
                                type="number", value=0.0, min=0.0, max=1.0, step=0.05, # Default 0.0 as backend doesn't use it yet
                                style={'width':'150px', 'verticalAlign':'middle'})
                    ], style={'marginBottom':'8px', 'textAlign':'left'}),
                    # Note: Adding more layers would require significant backend LSTMModel redesign.

                # --- Compilation / Training ---
                    html.Div([
                    html.Label("Time Steps (Sequence Len):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'time_steps'},
                                type="number", value=10, min=2, step=1, # Default from backend
                                style={'width':'150px', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                    html.Div([
                    html.Label("Optimizer:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                    dcc.Dropdown(id={**setting_id_base, 'param': 'optimizer'}, # Use 'optimizer' key like in backend
                                    options=[{'label': opt, 'value': opt} for opt in ['adam', 'rmsprop', 'sgd']],
                                    value='adam', clearable=False, # Default from backend
                                    style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                    html.Div([
                    html.Label("Learning Rate (float):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'learning_rate'},
                                type="number", value=0.001, step=0.0001, # Default from backend
                                style={'width':'150px', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                    html.Div([
                    html.Label("Loss Function:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                    dcc.Dropdown(id={**setting_id_base, 'param': 'loss'},
                                    options=[{'label': loss, 'value': loss} for loss in ['mse', 'mae']],
                                    value='mse', clearable=False, # Default from backend
                                    style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                html.Div([
                    html.Label("Training Epochs (int):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'epochs'},
                                type="number", value=10, min=1, step=1, # Default from backend
                                style={'width':'150px', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                    html.Div([
                    html.Label("Batch Size (int):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'batch_size'},
                                type="number", value=256, min=1, step=1, # Default from backend
                                style={'width':'150px', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
            ])
        elif selected_model == "svm":
            settings_children.extend([
                # --- Autoencoder Settings ---
                html.H6("Autoencoder Parameters:", style={'color':'#cccccc', 'marginTop':'10px', 'marginBottom': '8px', 'textAlign':'left'}),
                html.Div([
                    html.Label("Encoding Dimension (int):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'encoding_dim'},
                                type="number", value=10, min=2, step=1, # Default 10
                                style={'width':'150px', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                html.Div([
                    html.Label("AE Hidden Activation:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                    dcc.Dropdown(id={**setting_id_base, 'param': 'ae_activation'},
                                    options=[{'label': act, 'value': act} for act in ['relu', 'tanh', 'sigmoid', 'elu']],
                                    value='relu', clearable=False,
                                    style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                    html.Div([
                    html.Label("AE Output Activation:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                    dcc.Dropdown(id={**setting_id_base, 'param': 'ae_output_activation'},
                                    options=[{'label': act, 'value': act} for act in ['linear', 'sigmoid']], # Linear for StandardScaler, Sigmoid for MinMaxScaler
                                    value='linear', clearable=False,
                                    style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                html.Div([
                    html.Label("AE Optimizer:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                    dcc.Dropdown(id={**setting_id_base, 'param': 'optimizer'},
                                    options=[{'label': opt, 'value': opt} for opt in ['adam', 'rmsprop', 'sgd']],
                                    value='adam', clearable=False,
                                    style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                html.Div([
                    html.Label("AE Learning Rate (float):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'learning_rate'},
                                type="number", value=0.001, step=0.0001,
                                style={'width':'150px', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                html.Div([
                    html.Label("AE Loss Function:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                    dcc.Dropdown(id={**setting_id_base, 'param': 'loss'},
                                    options=[{'label': loss, 'value': loss} for loss in ['mse', 'mae']],
                                    value='mse', clearable=False,
                                    style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                    html.Div([
                    html.Label("AE Training Epochs (int):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'epochs'},
                                type="number", value=10, min=1, step=1, # Default 10
                                style={'width':'150px', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                    html.Div([
                    html.Label("AE Batch Size (int):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'batch_size'},
                                type="number", value=32, min=1, step=1, # Default 32
                                style={'width':'150px', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # --- OneClassSVM Settings ---
                html.H6("OneClassSVM Parameters:", style={'color':'#cccccc', 'marginTop':'20px', 'marginBottom': '8px', 'textAlign':'left'}),
                # Kernel (already present, adjusted style)
                html.Div([
                    html.Label("SVM Kernel:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                    dcc.Dropdown(id={**setting_id_base, 'param': 'svm_kernel'}, # Renamed param key
                                    options=[{'label': k, 'value': k} for k in ['rbf', 'linear', 'poly', 'sigmoid']],
                                    value='rbf', clearable=False,
                                    style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                # Nu (New)
                html.Div([
                    html.Label("SVM Nu (float 0-1):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'svm_nu'}, # Renamed param key
                                type="number", value=0.1, min=0.0, max=1.0, step=0.01, # Default 0.1
                                style={'width':'150px', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                # Gamma (already present, adjusted style)
                html.Div([
                    html.Label("SVM Gamma ('scale', 'auto', float):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'svm_gamma'}, # Renamed param key
                                type="text", value='scale', placeholder="'scale', 'auto' or float",
                                style={'width':'150px', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                # Degree (already present, adjusted style)
                html.Div([
                    html.Label("SVM Degree (int>=1, for Poly):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'svm_degree'}, # Renamed param key
                                type="number", value=3, min=1, step=1,
                                style={'width':'150px', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                # Coef0 (New)
                    html.Div([
                    html.Label("SVM Coef0 (float, Poly/Sigmoid):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'coef0'}, # Standard param name
                                type="number", value=0.0, step=0.1, # Default 0.0
                                style={'width':'150px', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                    # Shrinking (New)
                html.Div([
                    html.Label("SVM Shrinking Heuristic:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                    dcc.Dropdown(id={**setting_id_base, 'param': 'shrinking'}, # Standard param name
                                    options=[{'label': 'True', 'value': True}, {'label': 'False', 'value': False}],
                                    value=True, # Default is True
                                    clearable=False,
                                    style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                    # Tol (New)
                    html.Div([
                    html.Label("SVM Tolerance (float):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'tol'}, # Standard param name
                                type="number", value=1e-3, step=1e-4,
                                style={'width':'150px', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                    # Max Iter (New)
                    html.Div([
                    html.Label("SVM Max Iterations (int):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'max_iter'}, # Standard param name
                                type="number", placeholder="-1 (no limit)", step=1, min=-1, # Default -1
                                style={'width':'150px', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # Note: Regularization (C) is not a parameter for OneClassSVM, Nu is used instead. Removed 'C' input.
            ])
        elif selected_model == "isolation_forest":
            settings_children.extend([
                # --- Number of Estimators ---
                html.Div([
                    html.Label("Num Estimators (n_estimators):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'n_estimators'},
                                type="number", value=100, min=10, step=10, # Default 100
                                style={'width':'150px', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # --- Contamination ---
                html.Div([
                    html.Label("Contamination ('auto' or float):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'contamination'},
                                type="text", value='auto', placeholder="'auto' or float (0-0.5)", # Text allows 'auto' or float
                                style={'width':'150px', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # --- Max Samples ---
                html.Div([
                    html.Label("Max Samples ('auto', int, float):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'max_samples'},
                                type="text", value='auto', placeholder="'auto', int or float", # Text allows 'auto', int, or float
                                style={'width':'150px', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # --- Max Features ---
                html.Div([
                    html.Label("Max Features (float 0-1):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'max_features'},
                                type="number", value=1.0, min=0.0, max=1.0, step=0.05, # Float proportion
                                style={'width':'150px', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # --- Bootstrap ---
                html.Div([
                    html.Label("Bootstrap Samples:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Dropdown(id={**setting_id_base, 'param': 'bootstrap'},
                                    options=[{'label': 'False', 'value': False}, {'label': 'True', 'value': True}],
                                    value=False, # Default is False
                                    clearable=False,
                                    style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # --- Random State ---
                html.Div([
                    html.Label("Random State (int):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'random_state'},
                                type="number", placeholder="None", step=1, # Allow None (empty) or int
                                style={'width':'150px', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # Add more settings if needed...
            ])
        elif selected_model == "decision_tree":
            settings_children.extend([
                # --- Criterion ---
                html.Div([
                    html.Label("Criterion:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Dropdown(id={**setting_id_base, 'param': 'criterion'},
                                    options=[{'label': c, 'value': c} for c in ['gini', 'entropy', 'log_loss']], # Added log_loss
                                    value='gini', clearable=False,
                                    style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # --- Splitter ---
                html.Div([
                    html.Label("Splitter Strategy:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Dropdown(id={**setting_id_base, 'param': 'splitter'},
                                    options=[{'label': s, 'value': s} for s in ['best', 'random']],
                                    value='best', clearable=False,
                                    style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # --- Max Depth ---
                html.Div([
                    html.Label("Max Depth (int):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'max_depth'},
                                type="number", placeholder="None (unlimited)", step=1, min=1, # Ensure positive integer if set
                                style={'width':'150px', 'verticalAlign':'middle'}) # Allow None (empty)
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # --- Min Samples Split ---
                html.Div([
                    html.Label("Min Samples Split (int>=2):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'min_samples_split'},
                                type="number", value=2, min=2, step=1, # Default is 2
                                style={'width':'150px', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                    # --- Min Samples Leaf ---
                html.Div([
                    html.Label("Min Samples Leaf (int>=1):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'min_samples_leaf'},
                                type="number", value=1, min=1, step=1, # Default is 1
                                style={'width':'150px', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                    # --- Min Weight Fraction Leaf ---
                html.Div([
                    html.Label("Min Weight Fraction Leaf (float):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'min_weight_fraction_leaf'},
                                type="number", value=0.0, min=0.0, max=0.5, step=0.01, # Range 0.0 to 0.5
                                style={'width':'150px', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # --- Max Features ---
                html.Div([
                    html.Label("Max Features:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Dropdown(id={**setting_id_base, 'param': 'max_features'},
                                    # Common string options + None (all features)
                                    options=[{'label': mf, 'value': mf} for mf in ['sqrt', 'log2', 'None']],
                                    value='None', # Default is None (use all features)
                                    clearable=False,
                                    style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'}),
                    # Optional: Add a number input if user wants specific int/float?
                    # dcc.Input(id={**setting_id_base, 'param': 'max_features_num'}, type="number", placeholder="Or int/float", style={'width':'100px', 'marginLeft':'10px'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                    # --- Random State ---
                html.Div([
                    html.Label("Random State (int):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'random_state'},
                                type="number", placeholder="None", step=1,
                                style={'width':'150px', 'verticalAlign':'middle'}) # Allow None (empty) or int
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # --- Max Leaf Nodes ---
                html.Div([
                    html.Label("Max Leaf Nodes (int>=2):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'max_leaf_nodes'},
                                type="number", placeholder="None (unlimited)", step=1, min=2,
                                style={'width':'150px', 'verticalAlign':'middle'}) # Allow None (empty) or int >= 2
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # --- Min Impurity Decrease ---
                html.Div([
                    html.Label("Min Impurity Decrease (float):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'min_impurity_decrease'},
                                type="number", value=0.0, min=0.0, step=0.01, # Default is 0.0
                                style={'width':'150px', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # --- CCP Alpha (Pruning) ---
                html.Div([
                    html.Label("CCP Alpha (Pruning, float):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'ccp_alpha'},
                                type="number", value=0.0, min=0.0, step=0.01, # Default is 0.0
                                style={'width':'150px', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # Add more settings if needed...
                # Note: class_weight is handled in the backend DecisionTreeModel __init__ based on labels
            ])
        # Add elif blocks for other models (LSTM, SVM, Isolation Forest, Decision Tree)

        # If the selected model doesn't have specific settings defined here, hide the panel
        if len(settings_children) == 1: # Only contains the H5 title
            return [], {"display": "none"}

        return settings_children, panel_style

    # --- Callback to toggle visibility of XAI options panel ---
    @app.callback(
        Output("xai-options-div", "style"),
        Input("xai-check", "value")
    )
    def toggle_xai_panel(selected_xai):
        if "use_xai" in selected_xai:
             return {"display": "block", "marginTop": "10px", "textAlign": "center"}
        return {"display": "none"}
    
    # --- Callback to populate columns for Injection Dropdown AND Label Dropdown ---
    @app.callback(
        Output("injection-column-dropdown", "options"),
        Output("label-column-dropdown", "options"),
        Output("label-column-dropdown", "value"),
        Input("dataset-dropdown", "value"),
        prevent_initial_call=True
    )
    def update_column_dropdown(selected_dataset):
        print(f"--- update_column_dropdown callback triggered ---") # Start marker
        print(f"Selected Dataset: {selected_dataset!r}") # Use !r to clearly see None vs ""

        # Check if a dataset is actually selected
        if not selected_dataset:
            print("No dataset selected. Returning empty options.")
            return [], [], None

        handler = get_handler()
        print(f"Handler object: {handler}")
        columns = [] # Initialize columns
        options = [] # Initialize options
        try:
            print(f"Calling handler.handle_get_dataset_columns for '{selected_dataset}'...")
            columns = handler.handle_get_dataset_columns(selected_dataset)
            # *** Check what the handler actually returned ***
            print(f"Handler returned columns: {columns} (Type: {type(columns)})")

            # Ensure columns is a list before proceeding
            if not isinstance(columns, list):
                print("Warning: Handler did not return a list for columns. Returning empty options.")
                return [], [], None

            # Create options list
            options = [{"label": col, "value": col} for col in columns]
            print(f"Generated options for dropdowns: {options}")

            # Check if options are empty after filtering
            if not options:
                print("Warning: No column options remaining after filtering.")

            # Return options for both dropdowns, reset value for label dropdown
            return options, options, None

        except Exception as e:
            print(f"!!! ERROR inside update_column_dropdown try block: {e}")
            import traceback
            traceback.print_exc() # Print the full error traceback to the server console
            return [], [], None # Return empty on error

    # --- Callback to generate dynamic XAI settings panel ---
    @app.callback(
        Output("xai-settings-panel", "children"),
        Input("xai-method-dropdown", "value"),
        State("dataset-dropdown", "value"),
    )
    def update_xai_settings_panel(selected_xai_methods, selected_dataset):
        if not selected_xai_methods:
            return [] # Return empty if no method selected
        
        active_methods = [m for m in selected_xai_methods if m != 'none']
        if not active_methods:
             return []

        all_settings_children = [] # Initialize list to hold all settings components
        
        # --- Loop through each selected method ---
        for i, selected_xai_method in enumerate(active_methods):
            # Add a separator between methods if more than one is selected
            if i > 0:
                all_settings_children.append(html.Hr(style={'borderColor': '#555', 'margin': '20px 0'}))

            # Add heading for the current method
            method_settings = [html.H5(f"Settings for {selected_xai_method.upper()}:", style={'color':'#ffffff', 'marginTop':'15px', 'marginBottom': '10px'})]

            # --- Use Pattern-Matching IDs ---
            if selected_xai_method == "ShapExplainer":
                method_settings.extend([
                    html.Div([
                        html.Label("Indices to explain (n_explain_max):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                        dcc.Input(id={'type': 'xai-setting', 'method': 'ShapExplainer', 'param': 'n_explain_max'}, type="number", value=100, min=10, step=1, style={'width':'80px'})
                    ], style={'marginBottom':'8px'}),
                    html.Div([
                        html.Label("Num Samples (nsamples):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                        dcc.Input(id={'type': 'xai-setting', 'method': 'ShapExplainer', 'param': 'nsamples'}, type="number", value=100, min=10, step=1, style={'width':'80px'})
                    ], style={'marginBottom':'8px'}),
                    html.Div([
                        html.Label("K for Background Summary (k_summary):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                        dcc.Input(id={'type': 'xai-setting', 'method': 'ShapExplainer', 'param': 'k_summary'}, type="number", value=50, min=1, step=1, style={'width':'80px'})
                    ], style={'marginBottom':'8px'}),
                    html.Div([
                        html.Label("K for L1 Reg Features (l1_reg_k):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                        dcc.Input(id={'type': 'xai-setting', 'method': 'ShapExplainer', 'param': 'l1_reg_k'}, type="number", value=20, min=1, step=1, style={'width':'80px'})
                    ], style={'marginBottom':'8px'}),
                    html.Div([
                        html.Label("Explainer method:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                        dcc.Dropdown(id={'type': 'xai-setting', 'method': 'ShapExplainer', 'param': 'shap_method'}, options=[{'label': 'KernelShap (default)', 'value': 'kernel'},{'label': 'TreeShap', 'value': 'tree'},{'label': 'LinearShap', 'value': 'linear'},{'label': 'PartitionShap', 'value': 'partition'}], value='kernel', clearable=False, style={'width': '150px', 'display': 'inline-block', 'color': '#333'})
                    ], style={'marginBottom':'8px'})
                ])
            elif selected_xai_method == "LimeExplainer":
                method_settings.extend([
                    html.Div([
                        html.Label("Indices to explain (n_explain_max):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                        dcc.Input(id={'type': 'xai-setting', 'method': 'LimeExplainer', 'param': 'n_explain_max'}, type="number", value=10, min=1, step=1, style={'width':'80px'})
                    ], style={'marginBottom':'8px'}),
                    html.Div([
                        html.Label("Num Features to Explain:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                        dcc.Input(id={'type': 'xai-setting', 'method': 'LimeExplainer', 'param': 'num_features'}, type="number", value=15, min=1, step=1, style={'width':'80px'})
                    ], style={'marginBottom':'8px'}),
                    html.Div([
                        html.Label("Num Samples (Perturbations):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                        dcc.Input(id={'type': 'xai-setting', 'method': 'LimeExplainer', 'param': 'num_samples'}, type="number", value=1000, min=100, step=100, style={'width':'80px'})
                    ], style={'marginBottom':'8px'}),
                    html.Div([
                        html.Label("Kernel Width (kernel_width):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                        dcc.Input(id={'type': 'xai-setting', 'method': 'LimeExplainer', 'param': 'kernel_width'}, type="number", placeholder="LIME default", min=0.01, step=0.1, style={'width':'110px'})
                    ], style={'marginBottom':'8px'}),
                    html.Div([
                        html.Label("Feature Selection:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                        dcc.Dropdown(id={'type': 'xai-setting', 'method': 'LimeExplainer', 'param': 'feature_selection'}, options=[{'label': 'Auto', 'value': 'auto'},{'label': 'Highest Weights', 'value': 'highest_weights'},{'label': 'Forward Selection', 'value': 'forward_selection'},{'label': 'Lasso Path', 'value': 'lasso_path'},{'label': 'None', 'value': 'none'}], value='auto', clearable=False, style={'width': '180px', 'display': 'inline-block', 'color': '#333'})
                    ], style={'marginBottom':'8px'}),
                    html.Div([
                        html.Label("Discretize Continuous:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                        dcc.Dropdown(id={'type': 'xai-setting', 'method': 'LimeExplainer', 'param': 'discretize_continuous'}, options=[{'label': 'True', 'value': True}, {'label': 'False', 'value': False}], value=True, clearable=False, style={'width': '100px', 'display': 'inline-block', 'color': '#333'})
                    ], style={'marginBottom':'8px'}),
                    html.Div([
                        html.Label("Sample Around Instance:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                        dcc.Dropdown(id={'type': 'xai-setting', 'method': 'LimeExplainer', 'param': 'sample_around_instance'}, options=[{'label': 'True', 'value': True}, {'label': 'False', 'value': False}], value=True, clearable=False, style={'width': '100px', 'display': 'inline-block', 'color': '#333'})
                    ], style={'marginBottom':'8px'})
                ])
            elif selected_xai_method == "DiceExplainer":
                dice_specific_settings = [
                    html.Div([
                         html.Label("Indices to explain (n_explain_max):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                         dcc.Input(id={'type': 'xai-setting', 'method': 'DiceExplainer', 'param': 'n_explain_max'}, type="number", value=10, min=1, step=1, style={'width':'80px'})
                     ], style={'marginBottom':'8px'}),
                     html.Div([
                         html.Label("Num Counterfactuals (total_CFs):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                         dcc.Input(id={'type': 'xai-setting', 'method': 'DiceExplainer', 'param': 'total_CFs'}, type="number", value=5, min=1, step=1, style={'width':'80px'})
                     ], style={'marginBottom':'8px'}),
                     html.Div([
                         html.Label("Desired Class (desired_class):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                         dcc.Input(id={'type': 'xai-setting', 'method': 'DiceExplainer', 'param': 'desired_class'}, type="text", value="opposite", style={'width':'80px'})
                     ], style={'marginBottom':'8px'}),

                    # --- Dynamic Features to Vary Dropdown ---
                    html.Div([
                        html.Label("Features to vary (features_to_vary):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                        dcc.Dropdown(
                            # Use a specific index in MATCH perhaps? Or just ensure unique IDs if needed elsewhere
                            # For pattern matching state collection, the dict id is sufficient
                            id={'type': 'xai-setting', 'method': 'DiceExplainer', 'param': 'features_to_vary'},
                            options=[], # To be populated below
                            value=[],
                            multi=True,
                            placeholder="Select features (leave empty to vary all)",
                            style={'width': '90%', 'maxWidth':'500px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'}
                        )
                    ], id=f'dice-features-vary-div-{selected_xai_method}', # Make ID unique if needed elsewhere
                       style={'marginBottom':'8px'}), # Unique ID might not be needed if only accessed via pattern matching

                    html.Div([
                          html.Label("Backend (ML model framework):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                          dcc.Dropdown(id={'type': 'xai-setting', 'method': 'DiceExplainer', 'param': 'backend'}, options=[{'label': 'SciKit-Learn', 'value': 'sklearn'},{'label': 'Tensorflow 1', 'value': 'TF1'},{'label': 'Tensorflow 2', 'value': 'TF2'},{'label': 'PyTorch', 'value': 'pytorch'}], value='sklearn', clearable=False, style={'width': '150px', 'display': 'inline-block', 'color': '#333'})
                     ], style={'marginBottom':'8px'}),
                     html.Div([
                          html.Label("DiCE Method (dice_method):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                          dcc.Dropdown(id={'type': 'xai-setting', 'method': 'DiceExplainer', 'param': 'dice_method'}, options=[{'label': 'Random', 'value': 'random'},{'label': 'Genetic', 'value': 'genetic'},{'label': 'KD-Tree', 'value': 'kdtree'}], value='genetic', clearable=False, style={'width': '150px', 'display': 'inline-block', 'color': '#333'})
                     ], style={'marginBottom':'8px'})
                ]

                # --- Populate the features_to_vary dropdown options ---
                column_options = []
                if selected_dataset:
                    handler = get_handler()
                    try:
                        columns = handler.handle_get_dataset_columns(selected_dataset)
                        if isinstance(columns, list):
                            column_options = [{"label": col, "value": col} for col in columns]
                        else:
                            print(f"Warning: Handler did not return list for columns: {columns}")
                    except Exception as e:
                        print(f"!!! ERROR fetching columns for DiceExplainer: {e}")
                        traceback.print_exc()

                # Find the dropdown within dice_specific_settings and assign options
                # This assumes the dropdown is the second child of the Div with label "Features to vary..."
                for component in dice_specific_settings:
                     if isinstance(component, html.Div) and component.children and isinstance(component.children[0], html.Label):
                          if component.children[0].children == "Features to vary (features_to_vary):":
                               if len(component.children) > 1 and isinstance(component.children[1], dcc.Dropdown):
                                    component.children[1].options = column_options
                                    break

                method_settings.extend(dice_specific_settings)
            # Add elif for other methods
            # Append the generated settings for this method to the main list
            all_settings_children.extend(method_settings)

        return all_settings_children

    # --- Callback to toggle Speedup input based on mode ---
    @app.callback(
        Output("speedup-input-div", "style"),
        Input("mode-selection", "value")
    )
    def toggle_speedup_input(selected_mode):
        if selected_mode == "stream":
             return {"display": "block", "marginTop": "10px", "textAlign": "center"}
        return {"display": "none"}

    # --- Existing Callbacks for Active Jobs and Confirmation ---
    @app.callback(
        Output("active-jobs-section", "style"),
        Input("active-jobs-list", "children")
    )
    def toggle_active_jobs_section(children):
        # Show the active jobs section if there are active jobs
        if children == "No active jobs found.":
            return {"display": "none"}
        return {"display": "block", "marginTop": "30px"}
        
    # Callback to display confirmation box
    @app.callback(
        Output({"type": "confirm-box", "index": MATCH}, "displayed"),
        Input({"type": "remove-dataset-btn", "index": MATCH}, "n_clicks")
    )
    def display_confirm(value):
        return True if value else False

    # Callback to manage active jobs
    @app.callback(
        [Output("active-jobs-list", "children")],
        [Input("job-interval", "n_intervals"), Input({"type": "confirm-box", "index": ALL}, "submit_n_clicks")],
        [State("active-jobs-json", "data")],
        
    )
    def manage_and_remove_active_jobs(children, submit_n_clicks, active_jobs_json):
        ctx = callback_context

        triggered_id = ctx.triggered_id

        if triggered_id == None:
            return no_update

        handler = get_handler()

        if triggered_id != "job-interval":
            job = triggered_id["index"]
            handler.handle_cancel_job(job)

        active_jobs = json.loads(handler.handle_get_running())
        active_jobs = active_jobs["running"]
        jobs_json = json.dumps(active_jobs)

        if jobs_json == active_jobs_json:
            return no_update
        
        return create_active_jobs(active_jobs)

    # --- start_job_handler ---
    @app.callback(
        [Output("popup", "style"), Output("popup-interval", "disabled"), Output("popup", "children")],
        [Input("start-job-btn", "n_clicks"), Input("popup-interval", "n_intervals")],
        [
            State("dataset-dropdown", "value"), State("detection-model-dropdown", "value"),
            State("mode-selection", "value"), State("name-input", "value"),
            State("injection-method-dropdown", "value"), State("timestamp-input", "value"),
            State("magnitude-input", "value"), State("percentage-input", "value"),
            State("duration-input", "value"), State("injection-column-dropdown", "value"),
            State("injection-check", "value"), State("speedup-input", "value"),
            State("popup", "style"),
            # Labeled States
            State("labeled-check", "value"), State("label-column-dropdown", "value"),
            # --- XAI States (MODIFIED) ---
            State("xai-check", "value"),
            State("xai-method-dropdown", "value"), # Receives a LIST now
            # --- Pattern-Matching State for ALL XAI settings ---
            State({'type': 'xai-setting', 'method': ALL, 'param': ALL}, 'value'),
            State({'type': 'xai-setting', 'method': ALL, 'param': ALL}, 'id'),
            # --- NEW: Add ML Settings States ---
            State({'type': 'ml-setting', 'model': ALL, 'param': ALL}, 'value'),
            State({'type': 'ml-setting', 'model': ALL, 'param': ALL}, 'id'),
        ]
    )
    def start_job_handler(
            n_clicks, n_intervals,
            selected_dataset, selected_detection_model, selected_mode, job_name,
            selected_injection_method, timestamp, magnitude, percentage, duration,
            injection_columns, inj_check, speedup, style,
            labeled_check_val, selected_label_col,
            # --- ARGS for pattern-matching states ---
            xai_check_val,
            selected_xai_methods,
            xai_settings_values,
            xai_settings_ids,
            # --- NEW: Add ML Settings Args ---
            ml_settings_values,
            ml_settings_ids
            ):
        handler = get_handler()
        children = "Job submission processed."
        style_copy = style.copy()

        ctx = callback_context
        if not ctx.triggered or ctx.triggered[0]['prop_id'] != 'start-job-btn.n_clicks':
            if ctx.triggered and ctx.triggered[0]['prop_id'] == 'popup-interval.n_intervals':
                style_copy.update({"display": "none"})
                return style_copy, True, children
            return no_update, no_update, no_update

        trigger = ctx.triggered[0]["prop_id"]

        if trigger == "start-job-btn.n_clicks":
            # --- Basic Validation ---
            if not selected_dataset:
                style_copy.update({"backgroundColor": "#e74c3c", "display": "block"})
                return style_copy, False, "Please select a dataset."
            if not selected_detection_model:
                 style_copy.update({"backgroundColor": "#e74c3c", "display": "block"})
                 return style_copy, False, "Please select a detection model."
            if not job_name:
                style_copy.update({"backgroundColor": "#e74c3c", "display": "block"})
                return style_copy, False, "Job name cannot be empty."

            response = handler.check_name(job_name)
            if response != "success":
                 style_copy.update({"backgroundColor": "#e74c3c", "display": "block"})
                 return style_copy, False, "Job name already exists!"

            # --- Process Labeled Data Info ---
            is_labeled = "is_labeled" in labeled_check_val
            label_col_to_pass = None
            if is_labeled:
                if not selected_label_col:
                    style_copy.update({"backgroundColor": "#e74c3c", "display": "block"})
                    return style_copy, False, "Please select the label column for the labeled dataset."
                label_col_to_pass = selected_label_col
            # --- End Labeled Data Info ---

            # --- Process Injection Info ---
            inj_params_list = None # Backend expects list or None
            if "use_injection" in inj_check:
                # Add validation for injection params if needed
                if not selected_injection_method or selected_injection_method == "None":
                     style_copy.update({"backgroundColor": "#e74c3c", "display": "block"})
                     return style_copy, False, "Please select an injection method."
                if not timestamp:
                     style_copy.update({"backgroundColor": "#e74c3c", "display": "block"})
                     return style_copy, False, "Please enter an injection timestamp."
                # Basic validation passed, create params dict
                inj_params = {
                    "anomaly_type": selected_injection_method, "timestamp": str(timestamp),
                    "magnitude": str(magnitude if magnitude is not None else 1), # Use default if None
                    "percentage": str(percentage if percentage is not None else 0), # Default if None
                    "duration": str(duration if duration else '0s'), # Default if None/empty
                    "columns": injection_columns if injection_columns else [] # Use empty list if None
                }
                inj_params_list = [inj_params] # Backend expects a list
            # --- End Injection Info ---

            # --- Process XAI Info ---
            use_xai = "use_xai" in xai_check_val
        xai_params_list = None # Changed from xai_params to list

        if use_xai:
            active_methods = [m for m in selected_xai_methods if m != 'none'] if selected_xai_methods else []
            if not active_methods:
                style_copy.update({"backgroundColor": "#e74c3c", "display": "block"})
                return style_copy, False, "Please select at least one XAI method if 'Use Explainability' is checked."

            # --- Parse ALL Pattern-Matching State results into a structured dict ---
            # all_parsed_settings = { 'ShapExplainer': {'param1': val1, ...}, 'DiceExplainer': {'paramA': valA,...} }
            all_parsed_settings = {}
            print(f"DEBUG: Received XAI settings IDs: {xai_settings_ids}")
            print(f"DEBUG: Received XAI settings Values: {xai_settings_values}")

            for id_dict, value in zip(xai_settings_ids, xai_settings_values):
                method_name = id_dict['method']
                param_name = id_dict['param']

                # Only process settings for methods that are actually selected
                if method_name not in active_methods:
                    continue # Skip settings for methods not currently selected

                # Initialize dict for the method if it doesn't exist
                if method_name not in all_parsed_settings:
                    all_parsed_settings[method_name] = {}

                # Handle None values if necessary (user cleared input)
                # (Add specific default logic here if needed, similar to previous single-method version)
                if value is None and param_name not in ['features_to_vary', 'kernel_width']: # Allow None/empty for these
                     print(f"Warning: XAI setting '{param_name}' for method '{method_name}' has None value. Check defaults.")
                     # Example default assignment:
                     # if method_name == 'LimeExplainer' and param_name == 'num_samples': value = 1000

                # Special handling / Type conversion if needed
                if param_name == 'features_to_vary' and value == []:
                    print(f"DEBUG: features_to_vary for {method_name} is empty list. Backend might default to 'all'.")

                # Store the value
                all_parsed_settings[method_name][param_name] = value

            print(f"DEBUG: Parsed all XAI settings: {all_parsed_settings}")

            # --- Construct the final list of XAI params for the backend ---
            xai_params_list = []
            for method_name in active_methods:
                if method_name in all_parsed_settings:
                    current_settings = all_parsed_settings[method_name]

                    # Perform any key renaming needed for the backend *for this specific method*
                    if method_name == "ShapExplainer" and "l1_reg_k" in current_settings:
                        current_settings["l1_reg_k_features"] = current_settings.pop("l1_reg_k")

                    xai_params_list.append({
                        "method": method_name,
                        "settings": current_settings
                    })
                else:
                    # This case might happen if a method was selected but somehow its settings weren't rendered/parsed
                    print(f"Warning: No settings found/parsed for selected method: {method_name}")
                    # Decide how to handle: skip, add with empty settings, or error out?
                    # Example: Add with empty settings
                    # xai_params_list.append({"method": method_name, "settings": {}})

            if not xai_params_list: # If loop finished but list is empty (e.g., due to warnings/skips)
                 print("Error: Could not construct XAI parameters for selected methods.")
                 # Handle error appropriately

            print(f"DEBUG: Constructed xai_params_list for backend: {xai_params_list}")
        # --- End XAI Info ---

        # --- NEW: Process ML Model Settings ---
        ml_params_dict = {}
        if selected_detection_model: # Only parse if a model is selected
            print(f"DEBUG: Received ML settings IDs: {ml_settings_ids}")
            print(f"DEBUG: Received ML settings Values: {ml_settings_values}")

            for id_dict, value in zip(ml_settings_ids, ml_settings_values):
                # Ensure we only parse settings for the *currently selected* model
                if id_dict['model'] == selected_detection_model:
                    param_name = id_dict['param']
                    # Basic type conversion or validation could happen here if needed
                    # e.g., convert 'None' string placeholder for max_depth to None
                    if selected_detection_model == 'Decision Tree' and param_name == 'max_depth' and value is None:
                        actual_value = None
                    # e.g., handle 'auto' or float for contamination/max_samples
                    elif selected_detection_model == 'Isolation Forest' and param_name in ['contamination', 'max_samples'] and isinstance(value, str) and value.lower() != 'auto':
                        try:
                            actual_value = float(value)
                        except ValueError:
                            print(f"Warning: Invalid float value '{value}' for {param_name}. Using default.")
                            actual_value = 'auto' # Or keep the string, depending on backend expectation
                    else:
                        actual_value = value

                    ml_params_dict[param_name] = actual_value
                    print(f"DEBUG: Parsed ML Setting: {param_name} = {actual_value}")

            print(f"DEBUG: Constructed ml_params_dict for backend: {ml_params_dict}")

        # --- Call Backend Handler ---
        try:
            print(f"Sending job with parameters:")
            print(f"  Mode: {selected_mode}")
            print(f"  Dataset: {selected_dataset}")
            print(f"  Detection Model: {selected_detection_model}")
            print(f"  Job Name: {job_name}")
            print(f"  Label Column: {label_col_to_pass}")
            print(f"  XAI Params: {xai_params_list}")
            print(f"  Injection Params: {inj_params_list}")
            print(f"  ML Model Params: {ml_params_dict}") # Print new params
            sys.stdout.flush()

            # --- Modify backend call signatures to include ml_params_dict ---
            # You will need to update handle_run_batch and handle_run_stream
            # in your FrontendHandler and the corresponding backend methods.
            model_params_to_pass = ml_params_dict if ml_params_dict else None # Pass None if empty

            if selected_mode == "batch":
                response = handler.handle_run_batch(
                    selected_dataset, selected_detection_model, job_name,
                    label_column=label_col_to_pass, xai_params=xai_params_list, inj_params=inj_params_list,
                    model_params=model_params_to_pass # NEW ARG
                )
            else: # stream
                response = handler.handle_run_stream(
                    selected_dataset, selected_detection_model, job_name, speedup,
                    label_column=label_col_to_pass, xai_params=xai_params_list, inj_params=inj_params_list,
                    model_params=model_params_to_pass # NEW ARG
                )
            if response == "success":
                style_copy.update({"backgroundColor": "#4CAF50", "display": "block"})
                children = f"Job '{job_name}' started successfully!"
            else:
                style_copy.update({"backgroundColor": "#e74c3c", "display": "block"})
                children = f"Backend error starting job: {response}"

        except Exception as e:
            print(f"Error calling backend handler: {e}")
            style_copy.update({"backgroundColor": "#e74c3c", "display": "block"})
            children = "Error communicating with backend."

        return style_copy, False, children # Show popup

def create_active_jobs(active_jobs):
    if len(active_jobs) == 0:
        return ["No active jobs found."]
    job_divs = []
    GRAFANA_URL = f"http://localhost:{os.getenv('GRAFANA_PORT')}"
    for job in active_jobs:
        new_div = html.Div([
            dcc.ConfirmDialog(
                id={"type": "confirm-box", "index": job["name"]},
                message=f'Are you sure you want to cancel the job {job["name"]}?',
                displayed=False,
            ),
            html.A(
                children=[job["name"]],
                # href=f'/{job["name"]}' if job["type"] == "stream" else f'/{job["name"]}?batch=True', #Old version with bokeh
                href=f'{GRAFANA_URL}/d/stream01/stream-jobs' if job["type"] == "stream" else f'{GRAFANA_URL}/d/batch01/batch-jobs', # New version with grafana
                style={"marginRight": "10px", "color": "#4CAF50", "textDecoration": "none", "fontWeight": "bold"}
            ),
            html.Button("Stop", id={"type": "remove-dataset-btn", "index": job["name"]}, n_clicks=0, style={
                "fontSize": "12px", "backgroundColor": "#e74c3c", "color": "#ffffff", "border": "none",
                "borderRadius": "5px", "padding": "5px", "marginLeft": "7px"
            })
        ])
        job_divs.append(new_div)

    return [job_divs]