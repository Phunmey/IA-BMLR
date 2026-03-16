"""
Utility functions for data processing and evaluation
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    balanced_accuracy_score, roc_auc_score, confusion_matrix, log_loss
)
from imblearn.metrics import geometric_mean_score


def load_dataset(dataset_config, stratifying=False, n_stratify=100, return_dataframe=False):
    target_col = dataset_config['target_column']
    
    if stratifying:
        print(f"Stratifying {n_stratify} samples.")
        df_full = pd.read_parquet(dataset_config['path'])
        df = stratified_sample_n(df_full, label=target_col, n=n_stratify, random_state=42)
    else:
        print("Loading full dataset")
        df = pd.read_parquet(dataset_config['path'])

    if dataset_config['feature_columns'] is None:
        feature_cols = [col for col in df.columns if col != target_col]
    else:
        feature_cols = dataset_config['feature_columns']

    X = df[feature_cols]
    y_original = df[target_col].values

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_original)

    # Create explicit mapping
    label_mapping = {
        'original_to_encoded': {orig: enc for orig, enc in zip(le.classes_, range(len(le.classes_)))},
        'encoded_to_original': {enc: orig for orig, enc in zip(le.classes_, range(len(le.classes_)))},
        'encoder': le,
    }

    if 'class_names' in dataset_config and dataset_config['class_names'] is not None:
        class_names = [str(label_mapping['encoded_to_original'][i]) for i in range(len(le.classes_))]
    else:
        class_names = [str(c) for c in le.classes_]

    if return_dataframe:
        return X, y, feature_cols, class_names, label_mapping
    else:
        return X.values, y, feature_cols, class_names, label_mapping


def stratified_sample_n(df, label, n, random_state=42):
    """
    Return ~n rows sampled from df with class proportions matching the full dataset.
    """
    rng = np.random.default_rng(random_state)
    counts = df[label].value_counts().sort_index()
    props = counts / counts.sum()

    alloc = (props * n).astype(int)
    remainder = n - alloc.sum()
    if remainder > 0:
        fracs = (props * n) - (props * n).astype(int)
        bump_classes = fracs.sort_values(ascending=False).index[:remainder]
        alloc.loc[bump_classes] += 1

    parts = []
    for cls, k in alloc.items():
        if k > 0:
            parts.append(df[df[label] == cls].sample(n=k, random_state=random_state))
    return pd.concat(parts, axis=0).sample(frac=1, random_state=random_state).reset_index(drop=True)


def preprocess_data(X, y, test_size=0.2, random_state=42, scale=True, feature_names=None, verbose=True):
    if not isinstance(X, pd.DataFrame):
        if feature_names is not None:
            X = pd.DataFrame(X, columns=feature_names)
        else:
            X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])

    feature_names = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    preprocessor_info = {
        'original_features': feature_names.copy(),
        'numeric_features': num_cols.copy(),
        'categorical_features': cat_cols.copy(),
        'scaler': None,
        'encoder': None,
    }

    X_train_num = X_train[num_cols].values.astype(float) if num_cols else None
    X_test_num = X_test[num_cols].values.astype(float) if num_cols else None

    if scale and X_train_num is not None and len(num_cols) > 0:
        scaler = StandardScaler()
        X_train_num = scaler.fit_transform(X_train_num)
        X_test_num = scaler.transform(X_test_num)
        preprocessor_info['scaler'] = scaler

    X_train_cat = None
    X_test_cat = None

    if cat_cols:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X_train_cat = encoder.fit_transform(X_train[cat_cols])
        X_test_cat = encoder.transform(X_test[cat_cols])
        preprocessor_info['encoder'] = encoder

        if verbose:
            print(f"One-hot encoded {len(cat_cols)} features to {X_train_cat.shape[1]} columns")

    if X_train_num is not None and X_train_cat is not None:
        X_train_final = np.hstack([X_train_num, X_train_cat])
        X_test_final = np.hstack([X_test_num, X_test_cat])
    elif X_train_num is not None:
        X_train_final = X_train_num
        X_test_final = X_test_num
    elif X_train_cat is not None:
        X_train_final = X_train_cat
        X_test_final = X_test_cat
    else:
        raise ValueError("No features remaining after preprocessing!")

    return X_train_final, X_test_final, y_train, y_test, preprocessor_info


def preprocess_cv_fold(X_train_df, X_test_df, scale=True, verbose=False):
    """
    Preprocess a single CV fold (no splitting, just scaling and encoding).
    """
    cat_cols = X_train_df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X_train_df.select_dtypes(include=[np.number]).columns.tolist()

    preprocessor_info = {
        'numeric_features': num_cols.copy(),
        'categorical_features': cat_cols.copy(),
        'scaler': None,
        'encoder': None,
    }

    # Process numerical columns
    X_train_num = X_train_df[num_cols].values.astype(float) if num_cols else None
    X_test_num = X_test_df[num_cols].values.astype(float) if num_cols else None

    if scale and X_train_num is not None and len(num_cols) > 0:
        scaler = StandardScaler()
        X_train_num = scaler.fit_transform(X_train_num)
        X_test_num = scaler.transform(X_test_num)
        preprocessor_info['scaler'] = scaler

    # Process categorical columns
    X_train_cat = None
    X_test_cat = None

    if cat_cols:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X_train_cat = encoder.fit_transform(X_train_df[cat_cols])
        X_test_cat = encoder.transform(X_test_df[cat_cols])
        preprocessor_info['encoder'] = encoder

        if verbose:
            print(f"One-hot encoded {len(cat_cols)} features to {X_train_cat.shape[1]} columns")

    # Combine
    if X_train_num is not None and X_train_cat is not None:
        X_train = np.hstack([X_train_num, X_train_cat])
        X_test = np.hstack([X_test_num, X_test_cat])
    elif X_train_num is not None:
        X_train = X_train_num
        X_test = X_test_num
    elif X_train_cat is not None:
        X_train = X_train_cat
        X_test = X_test_cat
    else:
        raise ValueError("No features remaining after preprocessing!")

    return X_train, X_test, preprocessor_info


def create_cv_splits_indices(X, y, n_splits=5, random_state=42):
    """
    Create stratified K-fold splits (yields indices only).
    """
    y = np.asarray(y)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold_num, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        yield fold_num, train_idx, test_idx



def compute_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Compute comprehensive evaluation metrics.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'g_mean': geometric_mean_score(y_true, y_pred, average='multiclass'),
    }
    #
    # try:
    #     g_mean_val = geometric_mean_score(y_true, y_pred, average='multiclass')
    #     metrics['g_mean'] = g_mean_val
    # except Exception as e:
    #     metrics['g_mean'] = 0.0
    #
    # per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    # metrics['per_class_recall'] = per_class_recall.tolist()
    #
    # # If G-Mean is 0 due to a missing class, compute a "soft" G-Mean
    # # that uses a small epsilon instead of 0 (for comparison purposes only)
    # if metrics['g_mean'] == 0 and len(per_class_recall) > 0:
    #     # Add small epsilon to zero recalls for soft G-Mean
    #     eps = 1e-6
    #     soft_recalls = np.maximum(per_class_recall, eps)
    #     metrics['g_mean_soft'] = float(np.prod(soft_recalls) ** (1.0 / len(soft_recalls)))
    # else:
    #     metrics['g_mean_soft'] = metrics['g_mean']

    if y_pred_proba is not None:
        try:
            metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
            metrics['log_loss'] = log_loss(y_true, y_pred_proba)
        except:
            metrics['roc_auc_ovr'] = np.nan
            metrics['log_loss'] = np.nan

    # Per-class metrics
    metrics['per_class_precision'] = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
    metrics['per_class_recall'] = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
    metrics['per_class_f1'] = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()

    return metrics


def compute_class_statistics(y, class_names):
    """Compute class distribution statistics."""
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)

    stats = {
        'total_samples': total,
        'n_classes': len(unique),
        'class_distribution': {},
        'imbalance_ratio': None,
    }

    for cls, count in zip(unique, counts):
        cls_name = class_names[cls] if cls < len(class_names) else f"Class_{cls}"
        stats['class_distribution'][cls_name] = {'count': int(count), 'percentage': float(count / total * 100)}

    stats['imbalance_ratio'] = (counts.max() / counts.min())

    return stats


# def create_cv_splits(X, y, n_splits=5, random_state=42):
#     """Create stratified K-fold splits."""
#     X = np.asarray(X)
#     y = np.asarray(y)
#
#     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
#     for fold_num, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
#         X_train, X_test = X[train_idx], X[test_idx]
#         y_train, y_test = y[train_idx], y[test_idx]
#
#         yield fold_num, X_train, X_test, y_train, y_test


def aggregate_cv_results(fold_results):
    """
    Aggregate metrics across CV folds.
    """
    if not fold_results:
        return {}

    scalar_metrics = ['g_mean', 'balanced_accuracy', 'accuracy', 'f1', 'precision', 'recall', 'roc_auc_ovr', 'log_loss']

    aggregated = {}

    for metric in scalar_metrics:
        values = []
        for result in fold_results:
            val = result.get(metric)
            if val is not None and not np.isnan(val):
                values.append(val)

        if values:
            aggregated[f'{metric}_mean'] = np.mean(values)
            aggregated[f'{metric}_std'] = np.std(values)
        else:
            aggregated[f'{metric}_mean'] = np.nan
            aggregated[f'{metric}_std'] = np.nan

    return aggregated


def save_results(results, filepath):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def clean_dict(d):
        if isinstance(d, dict):
            keys_to_remove = {}
            for key, value in d.items():
                if key in ['model', 'model_initial', 'model_final', 'trace', 'convergence_summary']:
                    continue
                elif isinstance(value, dict):
                    keys_to_remove[key] = clean_dict(value)
                else:
                    keys_to_remove[key] = value
            return keys_to_remove
        return d

    results_clean = clean_dict(results)

    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {str(k): convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    serializable_results = convert_to_serializable(results_clean)

    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"Results saved to: {filepath}")


def load_results(filepath):
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        results = json.load(f)
    return results


def print_class_distribution(y, class_names, title="Class Distribution"):
    """Print formatted class distribution."""
    print(f"\n{title}")
    print("-" * 50)
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)

    for cls, count in zip(unique, counts):
        cls_name = class_names[cls] if cls < len(class_names) else f"Class_{cls}"
        pct = count / total * 100
        print(f"  {cls_name:<15}: {count:5d} ({pct:5.2f}%)")

    print(f" {'Total':<15}: {total:5d}")
    print(f" {'Imbalance Ratio':<15}: {counts.max() / counts.min():.2f}:1")
    print("-" * 50)


