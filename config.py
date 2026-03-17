"""
Configuration file for imbalanced classification experiments
"""

CONFIG = {'RANDOM_STATE': 42,
    'N_SPLITS': 5,
    'SINGLE_SPLIT': False,
    'TEST_SIZE': 0.2,

    'PYMC_SETTINGS': {'n_samples': 5000,  'n_tune': 5000,  'n_chains': 4,   'cores': 1,  'target_accept': 0.95},

    'UNCERTAINTY': {'compute': True,  'hdi_prob': 0.95},

    'KAPPA_SENSITIVITY': {'run': True, 'kappa_values': [5, 10, 15, 20]},

    'IA_BMLR': {'prior_sigma': 1.0, 'gamma_prior_sigma': 1.0, 'les_neighbors': 10},

    'STANDARD_BMLR': {'prior_sigma': 1.0},

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

    'RESULTS_DIR': './outputs/results',
    'PLOTS_DIR': './outputs/plots',
}

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
}
