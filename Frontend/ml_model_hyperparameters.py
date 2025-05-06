HYPERPARAMETER_DESCRIPTIONS = {
    "XGBoost": {
        "n_estimators": "Number of gradient boosted trees. Equivalent to the number of boosting rounds.",
        "learning_rate": "Step size shrinkage used in update to prevents overfitting. Range: (0, 1].",
        "max_depth": "Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit.",
        "min_child_weight": "Minimum sum of instance weight (hessian) needed in a child. Controls overfitting.",
        "gamma": "Minimum loss reduction required to make a further partition on a leaf node of the tree. Controls overfitting.",
        "subsample": "Fraction of samples to be used for fitting the individual base learners. Range: (0, 1].",
        "colsample_bytree": "Fraction of columns to be randomly sampled for each tree. Range: (0, 1].",
        "colsample_bylevel": "Fraction of columns for each split, in each level. Range: (0, 1].",
        "colsample_bynode": "Fraction of columns for each node (split). Range: (0, 1].",
        "reg_alpha": "L1 regularization term on weights. Increases model sparsity.",
        "reg_lambda": "L2 regularization term on weights. Makes model weights smaller.",
        "booster": "Which booster to use: 'gbtree' (tree-based), 'gblinear' (linear function), or 'dart'.",
        "random_state": "Random number seed for reproducibility.",
        "calibration_method": "Method ('isotonic' or 'sigmoid') used to calibrate predicted probabilities after base model training.",
        "n_jobs": "Number of CPU threads XGBoost will use. More = Faster but heavier load..",
        # Add others as needed
    },
    "lstm": {
        "units": "Dimensionality of the output space (number of units) in the LSTM layers.",
        "activation": "Activation function to use in LSTM layers (e.g., 'relu', 'tanh').",
        "dropout": "Fraction of the units to drop for the linear transformation of the inputs.",
        "recurrent_dropout": "Fraction of the units to drop for the linear transformation of the recurrent state.",
        "time_steps": "Length of the input sequences (lookback window).",
        "optimizer": "Algorithm used to update model weights (e.g., 'adam', 'rmsprop').",
        "learning_rate": "Controls the step size during optimization.",
        "loss": "Function to measure the error between predictions and reality (e.g., 'mse', 'mae').",
        "epochs": "Number of complete passes through the entire training dataset.",
        "batch_size": "Number of samples processed before the model is updated."
        # Add others as needed
    },
    "svm": {
        # Autoencoder Params
        "encoding_dim": "Dimensionality of the Autoencoder's compressed representation (latent space).",
        "ae_activation": "Activation function for the Autoencoder's hidden layer(s).",
        "ae_output_activation": "Activation function for the Autoencoder's final output layer ('linear' recommended for StandardScaler).",
        "optimizer": "Optimizer used for training the Autoencoder.",
        "learning_rate": "Learning rate for the Autoencoder's optimizer.",
        "loss": "Loss function used to train the Autoencoder (e.g., reconstruction error like 'mse').",
        "epochs": "Number of training epochs for the Autoencoder.",
        "batch_size": "Batch size used for training the Autoencoder.",
        # OneClassSVM Params
        "svm_kernel": "Specifies the kernel type to be used in the OneClassSVM algorithm ('rbf', 'linear', 'poly', 'sigmoid').",
        "svm_nu": "An upper bound on the fraction of training errors and a lower bound on the fraction of support vectors. Range: (0, 1].",
        "svm_gamma": "Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. 'scale' uses 1 / (n_features * X.var()), 'auto' uses 1 / n_features.",
        "svm_degree": "Degree of the polynomial kernel function ('poly'). Ignored by other kernels.",
        "coef0": "Independent term in kernel function. Only significant in 'poly' and 'sigmoid'.",
        "shrinking": "Whether to use the shrinking heuristic.",
        "tol": "Tolerance for stopping criterion.",
        "max_iter": "Hard limit on iterations within the solver, or -1 for no limit."
        # Add others as needed
    },
    "isolation_forest": {
        "n_estimators": "The number of base estimators (trees) in the ensemble.",
        "contamination": "The expected proportion of outliers in the data set. Used for threshold when 'predict' is used. 'auto' estimates it.",
        "max_samples": "The number of samples (int) or fraction (float) to draw from data to train each base estimator.",
        "max_features": "The number of features (int) or fraction (float) to draw from data to train each base estimator.",
        "bootstrap": "If True, individual trees are fit on random subsets of the training data sampled with replacement. If False, sampling without replacement.",
        "random_state": "Controls the pseudo-randomness of building trees and drawing samples."
        # Add others as needed
    },
    "decision_tree": {
        "criterion": "Function to measure the quality of a split ('gini', 'entropy', 'log_loss').",
        "splitter": "Strategy used to choose the split at each node ('best' or 'random').",
        "max_depth": "Maximum depth of the tree. If None, nodes are expanded until all leaves are pure or contain less than min_samples_split samples.",
        "min_samples_split": "Minimum number of samples required to split an internal node.",
        "min_samples_leaf": "Minimum number of samples required to be at a leaf node.",
        "min_weight_fraction_leaf": "Minimum weighted fraction of the sum total of weights required to be at a leaf node.",
        "max_features": "Number of features to consider when looking for the best split ('sqrt', 'log2', None=all).",
        "random_state": "Controls the randomness of the estimator (for splitter='random' and/or max_features<n_features).",
        "max_leaf_nodes": "Grow a tree with max_leaf_nodes in best-first fashion. If None, unlimited number of leaf nodes.",
        "min_impurity_decrease": "A node will be split if this split induces a decrease of the impurity greater than or equal to this value.",
        "ccp_alpha": "Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost complexity that is smaller than ccp_alpha will be chosen."
        # Add others as needed
    }
    # Add entries for other models if needed
}


# Example placement: Near the top of callbacks.py or in a separate file

XAI_METHOD_DESCRIPTIONS = {
    "ShapExplainer": {
        "description": "SHAP (SHapley Additive exPlanations) assigns each feature an importance value for a particular prediction based on cooperative game theory. It explains how much each feature contributes to pushing the model output from the base value (average model output over the training dataset) to the current prediction.",
        "capabilities": "Provides global and local explanations, theoretically sound (Shapley values), consistent, handles feature interactions to some extent.",
        "limitations": "Can be computationally expensive, especially KernelSHAP (model-agnostic version). TreeSHAP is faster but specific to tree models. Interpretation of interaction effects can be complex.",
        "parameters": {
            "n_explain_max": "Maximum number of instances (predictions) to explain. Should be larger since it gives a global explanation",
            "nsamples": "(KernelSHAP) Number of times to sample perturbations for each explanation. Higher values increase accuracy but also computation time.",
            "k_summary": "(KernelSHAP) Number of samples from the background dataset used to summarize it (e.g., using k-means).",
            "l1_reg_k": "(KernelSHAP) Number of features to select using L1 regularization (Lasso) when approximating Shapley values. Controls sparsity.",
            "shap_method": "Specifies the SHAP algorithm variant to use ('kernel', 'tree', 'linear', 'partition'). 'kernel' is model-agnostic but slower. 'tree' is optimized for tree-based models (like XGBoost, Decision Tree)."
        }
    },
    "LimeExplainer": {
        "description": "LIME (Local Interpretable Model-agnostic Explanations) explains individual predictions of any black-box model by learning a simpler, interpretable linear model locally around the prediction.",
        "capabilities": "Model-agnostic, provides intuitive local explanations (feature importance for a specific prediction), relatively easy to understand.",
        "limitations": "Explanations are local and may not represent global model behavior. Sensitive to hyperparameter choices (kernel width, number of samples). Explanation instability can occur. May struggle with highly non-linear interactions.",
        "parameters": {
            "n_explain_max": "Maximum number of instances (predictions) to explain.",
            "num_features": "Maximum number of features to include in the local explanation.",
            "num_samples": "Number of perturbed samples generated around the instance to train the local linear model.",
            "kernel_width": "Width of the kernel function used to weight perturbed samples based on proximity to the original instance. Smaller values focus more locally.",
            "feature_selection": "Method used to select features for the local explanation ('auto', 'highest_weights', 'forward_selection', 'lasso_path', 'none').",
            "discretize_continuous": "Whether to discretize continuous features for perturbation and explanation generation.",
            "sample_around_instance": "Whether to sample perturbations centered around the instance being explained."
        }
    },
    "DiceExplainer": {
        "description": "DiCE (Diverse Counterfactual Explanations) generates counterfactual examples, which are minimal changes to feature values that flip the model's prediction to a desired outcome (e.g., from 'anomaly' to 'normal'). It aims to provide diverse examples.",
        "capabilities": "Model-agnostic, provides actionable insights by showing what needs to change for a different outcome, generates multiple diverse counterfactuals.",
        "limitations": "Finding counterfactuals can be computationally expensive. Generated counterfactuals might not always be realistic or feasible. Primarily focused on 'what-if' scenarios rather than direct feature importance.",
        "parameters": {
            "n_explain_max": "Maximum number of instances (predictions) to find counterfactuals for.",
            "total_CFs": "Desired number of diverse counterfactual examples to generate per instance.",
            "desired_class": "The target prediction class for the counterfactuals (e.g., 'opposite' to flip the current prediction, or a specific class index like 0 or 1).",
            "features_to_vary": "List of feature names that are allowed to be changed when generating counterfactuals. If empty or not specified, DiCE typically considers all mutable features.",
            "backend": "Specifies the machine learning framework the model was built with ('sklearn', 'TF1', 'TF2', 'pytorch') to ensure compatibility.",
            "dice_method": "Algorithm used to generate counterfactuals ('random', 'genetic', 'kdtree'). 'genetic' often finds better counterfactuals but is slower."
        }
    }
    # Add entries for other XAI methods if you implement them
}