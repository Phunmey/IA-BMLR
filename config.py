"""
Configuration file for imbalanced classification experiments
"""

CONFIG = {'RANDOM_STATE': 42,
    'N_SPLITS': 5,
    'SINGLE_SPLIT': False,
    'TEST_SIZE': 0.2,

    # ============================================================
    # PYMC SETTINGS (MCMC)
    # ============================================================
    'PYMC_SETTINGS': {
        'n_samples': 5000,      # Posterior samples per chain
        'n_tune': 5000,         # Tuning samples per chain
        'n_chains': 4,          # Number of chains
        'cores': 1,             # CPU cores for parallel chain sampling.
                                # Set to the number of cores available on
                                # your cluster node. When cores == n_chains,
                                # all chains run simultaneously, giving the
                                # maximum wall-clock speedup.
        'target_accept': 0.95,  # Target acceptance rate
    },

    # ============================================================
    # UNCERTAINTY QUANTIFICATION SETTINGS
    # ============================================================
    # Controls whether the full predictive uncertainty summary
    # (HDI bounds, HDI width, predictive entropy) is computed and
    # saved alongside the standard performance metrics.
    # Adds modest overhead on top of sampling cost, but captures
    # all outputs needed to demonstrate the Bayesian advantage
    # over frequentist baselines (Reviewer C).
    'UNCERTAINTY': {
        'compute': True,        # Set False to skip HDI computation
        'hdi_prob': 0.95,       # Probability mass enclosed by HDI
    },

    # ============================================================
    # KAPPA SENSITIVITY SETTINGS  (Reviewer A minor comment)
    # ============================================================
    # Runs IA-BMLR under several kappa values on the first split
    # only (not full CV) to keep cost manageable. Results are
    # saved alongside the main CV results for each dataset.
    'KAPPA_SENSITIVITY': {
        'run': True,
        'kappa_values': [5, 10, 15, 20],
    },

    # ============================================================
    # IA-BMLR v2 SETTINGS (Weighted likelihood, no augmentation)
    # ============================================================
    # Features:
    # - Uniform priors: β_k ~ N(0, σ²I) for all k
    # - Weighted likelihood: w_i = w_class(y_i) × w_entropy(H_i)
    # - w_class(k) = N / (K × n_k)  [inverse frequency]
    # - w_entropy(i) = (1 + H_i)^γ  [entropy focus]
    'IA_BMLR': {
        'prior_sigma': 1.0,     # Prior std (uniform for all classes)
        'gamma_prior_sigma': 1.0,           # Entropy focus (0=ignore, higher=more focus)
        'les_neighbors': 10,    # k for Local Entropy Score
    },

    # ============================================================
    # STANDARD BMLR SETTINGS (Baseline - no weighting)
    # ============================================================
    'STANDARD_BMLR': {
        'prior_sigma': 1.0,
    },
    
    # SMOTE settings (for baseline)
    'SMOTE': {
        'k_neighbors': 5,
        'sampling_strategy': 'auto',  # Balance to majority class
    },
    
    # Baseline ML models
    'BASELINE_MODELS': {
        'multinomial_lr': {
            'max_iter': 10000,
            'solver': 'lbfgs',
            'class_weight': None,
            'random_state': 42,
        },
        'random_forest': {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'class_weight': None,
            'random_state': 42,
        },
        'xgboost': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'objective': 'multi:softmax',
            'random_state': 42,
        },
        'svm': {
            'kernel': 'rbf',
            'class_weight': None,
            'random_state': 42,
            'probability': True,
        }
    },
    
    # Metrics to compute
    'METRICS': [
        'accuracy', 
        'precision', 
        'recall', 
        'f1', 
        'balanced_accuracy', 
        'roc_auc_ovr', 
        'log_loss',
        'g_mean',
    ],
    
    # Results directory
    'RESULTS_DIR': './outputs/results',
    'PLOTS_DIR': './outputs/plots',
}

# Dataset configurations
DATASETS = {
    'hayesroth_data': {
        'path': r'./keel_clean_data/hayesroth.parquet',
        'target_column': 'class',
        'feature_columns': None,
        'class_names': ['1', '2', '3'],
    },
    'wine': {
        'path': r'./keel_clean_data/wine.parquet',
        'target_column': 'class',
        'feature_columns': None,
        'class_names': ['1', '2', '3'],
    },
    'lymphography_data': {
        'path': r'./keel_clean_data/lymphography.parquet',
        'target_column': "'class'",
        'feature_columns': None,
        'class_names': ['normal', 'metastases', 'malign_lymph', 'fibrosis'],
    },
    'glass_data': {
        'path': r'./keel_clean_data/glass.parquet',
        'target_column': 'typeGlass',
        'feature_columns': None,
        'class_names': ['1', '2', '3', '5', '6', '7'],
    },
    'newthyroid_data': {
        'path': r'./keel_clean_data/newthyroid.parquet',
        'target_column': 'class',
        'feature_columns': None,
        'class_names': ['normal', 'hyper', 'hypo'],
    },
    'ecoli_data': {
        'path': r'./keel_clean_data/ecoli.parquet',
        'target_column': 'class',
        'feature_columns': None,
        'class_names': ['cp', 'im', 'pp', 'imU', 'om', 'omL', 'imL', 'imS'],
    },
    'pageblocks_data': {
        'path': r'./keel_clean_data/pageblocks.parquet',
        'target_column': 'class',
        'feature_columns': None,
        'class_names': ['1', '2', '4', '5', '3'],
    },
    'contraceptive_data': {
        'path': r'./keel_clean_data/contraceptive.parquet',
        'target_column': 'class',
        'feature_columns': None,
        'class_names': ['1', '2', '3'],
    },
    'yeast': {
        'path': r'./keel_clean_data/yeast.parquet',
        'target_column': 'class',
        'feature_columns': None,
        'class_names': ['MIT', 'NUC', 'CYT', 'ME1', 'ME2', 'ME3', 'EXC', 'VAC', 'POX', 'ERL'],
    },
    # 'student_data': {
    #     'path': r'./keel_clean_data/student.parquet',
    #     'target_column': 'Status',
    #     'feature_columns': None,
    #     'class_names': ['Dropout', 'Enrolled', 'Graduate'],
    # },
}
