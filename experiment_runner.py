"""
Experiment runner for IA-BMLR revision.
"""

import os
import sys
import time
import argparse
import traceback
import numpy as np
import arviz as az

from xgboost import XGBClassifier
from config import CONFIG, DATASETS
from utils import (
    load_dataset, preprocess_cv_fold, save_results, aggregate_cv_results,
    print_class_distribution, compute_class_statistics,
    create_cv_splits_indices, compute_metrics
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from model_ia_bmlr import IABMLR
from model_standard_bmlr import StandardBMLR



def fit_baseline_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred      = model.predict(X_test)
    y_proba     = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    train_pred  = model.predict(X_train)
    train_proba = model.predict_proba(X_train) if hasattr(model, 'predict_proba') else None
    return {
        'train_metrics': compute_metrics(y_train, train_pred, train_proba),
        'test_metrics':  compute_metrics(y_test,  y_pred,     y_proba),
    }


def fit_all_baselines(X_train, y_train, X_test, y_test, config):
    results = {}
    baseline_config = config.get('BASELINE_MODELS', {})

    if 'multinomial_lr' in baseline_config:
        print("\n  Fitting Multinomial LR...")
        lr_config = baseline_config['multinomial_lr']
        model = LogisticRegression(
            max_iter=lr_config.get('max_iter', 10000), solver=lr_config.get('solver', 'lbfgs'),
            class_weight=lr_config.get('class_weight'), random_state=lr_config.get('random_state', 42))
        results['multinomial_lr'] = {
            **fit_baseline_model(model, X_train, y_train, X_test, y_test),
            'model': model, 'training_size': len(X_train), 'method': 'Multinomial LR'}

    if 'random_forest' in baseline_config:
        print("\n  Fitting Random Forest...")
        rf_config = baseline_config['random_forest']
        model = RandomForestClassifier(
            n_estimators=rf_config.get('n_estimators', 100), max_depth=rf_config.get('max_depth'),
            min_samples_split=rf_config.get('min_samples_split', 2),
            class_weight=rf_config.get('class_weight'), random_state=rf_config.get('random_state', 42))
        results['random_forest'] = {
            **fit_baseline_model(model, X_train, y_train, X_test, y_test),
            'model': model, 'training_size': len(X_train), 'method': 'Random Forest'}

    if 'svm' in baseline_config:
        print("\n  Fitting SVM...")
        svm_config = baseline_config['svm']
        model = SVC(kernel=svm_config.get('kernel', 'rbf'), class_weight=svm_config.get('class_weight'),
                random_state=svm_config.get('random_state', 42), probability=svm_config.get('probability', True))
        results['svm'] = {
            **fit_baseline_model(model, X_train, y_train, X_test, y_test),
            'model': model, 'training_size': len(X_train), 'method': 'SVM (RBF)'}

    if 'xgboost' in baseline_config:
        try:
            print("\n  Fitting XGBoost...")
            xgb_config = baseline_config['xgboost']
            model = XGBClassifier(
                n_estimators=xgb_config.get('n_estimators', 100), max_depth=xgb_config.get('max_depth', 6),
                learning_rate=xgb_config.get('learning_rate', 0.1), objective='multi:softmax',
                num_class=len(np.unique(y_train)),
                random_state=xgb_config.get('random_state', 42), eval_metric='mlogloss')
            results['xgboost'] = {
                **fit_baseline_model(model, X_train, y_train, X_test, y_test),
                'model': model, 'training_size': len(X_train), 'method': 'XGBoost'}
        except ImportError:
            print("  XGBoost not available, skipping...")

    return results


def _build_ia_bmlr(config, kappa_override=None):
    pymc_cfg = config.get('PYMC_SETTINGS', {})
    ia_cfg   = config.get('IA_BMLR', {})
    kappa    = kappa_override if kappa_override is not None else ia_cfg.get('les_neighbors', 10)
    return IABMLR(
        prior_sigma       = ia_cfg.get('prior_sigma', 1.0),
        gamma_prior_sigma = ia_cfg.get('gamma_prior_sigma', 1.0),
        les_neighbors     = kappa,
        n_samples         = pymc_cfg.get('n_samples', 2000),
        n_tune            = pymc_cfg.get('n_tune', 1000),
        n_chains          = pymc_cfg.get('n_chains', 2),
        cores             = pymc_cfg.get('cores', None),
        target_accept     = pymc_cfg.get('target_accept', 0.95),
    )


def fit_ia_bmlr_fold(X_train, y_train, X_test, y_test, config, verbose=False, kappa_override=None):
    unc_cfg  = config.get('UNCERTAINTY', {})
    do_unc   = unc_cfg.get('compute', True)
    hdi_prob = unc_cfg.get('hdi_prob', 0.95)

    model = _build_ia_bmlr(config, kappa_override=kappa_override)
    model.fit(X_train, y_train, verbose=verbose, save_trace=False)

    train_metrics, test_metrics, train_unc, test_unc = model.evaluate_train_test(X_test, y_test, compute_uncertainty=do_unc, hdi_prob=hdi_prob)

    result = {
        'model':         model,
        'train_metrics': train_metrics,
        'test_metrics':  test_metrics,
        'training_size': len(X_train),
        'method':        'IA-BMLR',
        'learned_gamma': model.learned_gamma,
    }

    if do_unc and test_unc is not None:
        result['test_uncertainty']  = _serialise_uncertainty(test_unc)
    if do_unc and train_unc is not None:
        result['train_uncertainty'] = _serialise_uncertainty(train_unc)

    return result


def fit_standard_bmlr_fold(X_train, y_train, X_test, y_test, config, verbose=False):
    unc_cfg  = config.get('UNCERTAINTY', {})
    do_unc   = unc_cfg.get('compute', True)
    hdi_prob = unc_cfg.get('hdi_prob', 0.95)

    pymc_cfg = config.get('PYMC_SETTINGS', {})
    bmlr_cfg = config.get('STANDARD_BMLR', {})

    model = StandardBMLR(
        prior_sigma   = bmlr_cfg.get('prior_sigma', 1.0),
        n_samples     = pymc_cfg.get('n_samples', 2000),
        n_tune        = pymc_cfg.get('n_tune', 1000),
        n_chains      = pymc_cfg.get('n_chains', 2),
        cores         = pymc_cfg.get('cores', None),
        target_accept = pymc_cfg.get('target_accept', 0.95),
    )
    model.fit(X_train, y_train, verbose=verbose, save_trace=False)

    train_metrics, test_metrics, train_unc, test_unc = model.evaluate_train_test(X_test, y_test, compute_uncertainty=do_unc, hdi_prob=hdi_prob)

    result = {
        'model':         model,
        'train_metrics': train_metrics,
        'test_metrics':  test_metrics,
        'training_size': len(X_train),
        'method':        'Standard BMLR',
    }

    if do_unc and test_unc is not None:
        result['test_uncertainty']  = _serialise_uncertainty(test_unc)
    if do_unc and train_unc is not None:
        result['train_uncertainty'] = _serialise_uncertainty(train_unc)

    return result


def _serialise_uncertainty(unc_dict):
    serialised = {}
    for key, val in unc_dict.items():
        serialised[key] = val.tolist() if isinstance(val, np.ndarray) else val

    if 'hdi_width' in unc_dict:
        w = unc_dict['hdi_width']                              # (N, K)
        serialised['mean_hdi_width_per_class']  = w.mean(axis=0).tolist()
        serialised['mean_hdi_width_overall']    = float(w.mean())

        if 'proba_mean' in unc_dict:
            pred_idx = np.argmax(unc_dict['proba_mean'], axis=1)
            pred_class_width = w[np.arange(len(pred_idx)), pred_idx]
            serialised['frac_uncertain_predictions'] = float((pred_class_width > 0.2).mean())

    if 'pred_entropy' in unc_dict:
        serialised['mean_pred_entropy'] = float(unc_dict['pred_entropy'].mean())

    return serialised


def _aggregate_uncertainty_across_folds(fold_uncertainty_list):
    if not fold_uncertainty_list:
        return {}

    scalar_keys = ['mean_hdi_width_overall', 'mean_pred_entropy', 'frac_uncertain_predictions']
    aggregated = {}

    for key in scalar_keys:
        vals = [f[key] for f in fold_uncertainty_list if key in f]
        if vals:
            aggregated[f'{key}_mean'] = float(np.mean(vals))
            aggregated[f'{key}_std']  = float(np.std(vals))

    per_class_vals = [f['mean_hdi_width_per_class'] for f in fold_uncertainty_list if 'mean_hdi_width_per_class' in f]
    if per_class_vals:
        arr = np.array(per_class_vals)              # (n_folds, K)
        aggregated['mean_hdi_width_per_class_mean'] = arr.mean(axis=0).tolist()
        aggregated['mean_hdi_width_per_class_std']  = arr.std(axis=0).tolist()

    return aggregated


def run_kappa_sensitivity(X_train, y_train, X_test, y_test, config, verbose=False):
    kappa_values = config.get('KAPPA_SENSITIVITY', {}).get('kappa_values', [5, 10, 15, 20])
    kappa_results = {}

    for kappa in kappa_values:
        print(f"\n    kappa sensitivity: kappa={kappa} ...")
        try:
            result = fit_ia_bmlr_fold(X_train, y_train, X_test, y_test, config, verbose=verbose, kappa_override=kappa)
            model = result['model']
            kappa_results[kappa] = {
                'test_metrics':  result['test_metrics'],
                'learned_gamma': result['learned_gamma'],
                'les_mean':      float(model.les_scores.mean()),
                'les_std':       float(model.les_scores.std()),
                'les_min':       float(model.les_scores.min()),
                'les_max':       float(model.les_scores.max()),
            }
            if 'test_uncertainty' in result:
                kappa_results[kappa]['uncertainty_summary'] = {k: v for k, v in result['test_uncertainty'].items() if not isinstance(v, list)}
            print(f" G-Mean={result['test_metrics'].get('g_mean', 0):.4f}, gamma={result['learned_gamma']:.4f}, LES mean={model.les_scores.mean():.4f}")
        except Exception as e:
            print(f"kappa={kappa} failed: {e}")
            traceback.print_exc()
            kappa_results[kappa] = {'error': str(e)}

    return kappa_results

class ExperimentRunner:
    def __init__(self, config=CONFIG):
        self.config  = config
        self.results = {}
        os.makedirs(config['RESULTS_DIR'], exist_ok=True)
        os.makedirs(config['PLOTS_DIR'],   exist_ok=True)

    def run_single_dataset(self, dataset_name, dataset_config):
        dataset_results_dir = os.path.join(self.config['RESULTS_DIR'], dataset_name)
        dataset_plots_dir   = os.path.join(self.config['PLOTS_DIR'],   dataset_name)
        os.makedirs(dataset_results_dir, exist_ok=True)
        os.makedirs(dataset_plots_dir,   exist_ok=True)

        old_results = self.config['RESULTS_DIR']
        old_plots   = self.config['PLOTS_DIR']
        self.config['RESULTS_DIR'] = dataset_results_dir
        self.config['PLOTS_DIR']   = dataset_plots_dir

        try:
            return self._run_dataset_experiments(dataset_name, dataset_config, dataset_results_dir, dataset_plots_dir)
        finally:
            self.config['RESULTS_DIR'] = old_results
            self.config['PLOTS_DIR']   = old_plots

    def _run_dataset_experiments(self, dataset_name, dataset_config, dataset_results_dir, dataset_plots_dir):
        print(f"\nStarted: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        X, y, feature_names, class_names, label_mapping = load_dataset(dataset_config, stratifying=False, return_dataframe=True)
        print_class_distribution(y, class_names, "Original Data")

        class_stats = compute_class_statistics(y, class_names)
        n_splits    = self.config.get('N_SPLITS', 5)

        unique, counts = np.unique(y, return_counts=True)
        if counts.min() < n_splits:
            n_splits = max(2, int(counts.min()))

        dataset_results = {
            'dataset_name':    dataset_name,
            'n_samples':       len(X),
            'n_features':      len(feature_names),
            'n_classes':       len(class_names),
            'class_names':     class_names,
            'feature_names':   feature_names,
            'class_statistics':class_stats,
            'n_splits':        n_splits,
            'cv_results':      {},
            'models':          {},
        }

        model_names  = ['multinomial_lr', 'random_forest', 'svm', 'xgboost', 'ia_bmlr', 'standard_bmlr']
        fold_results = {name: {'train': [], 'test': []} for name in model_names}
        unc_folds    = {'ia_bmlr': [], 'standard_bmlr': []}
        learned_gammas   = []
        last_fold_models = {}
        kappa_sensitivity_done = False

        for fold_num, train_idx, test_idx in create_cv_splits_indices(X, y, n_splits=n_splits, random_state=self.config['RANDOM_STATE']):
            X_train_df = X.iloc[train_idx].reset_index(drop=True)
            X_test_df  = X.iloc[test_idx].reset_index(drop=True)
            y_train    = y[train_idx]
            y_test     = y[test_idx]

            X_train, X_test, _ = preprocess_cv_fold(X_train_df, X_test_df, scale=True, verbose=(fold_num == 1))

            print("\n  BASELINE CLASSIFIERS")
            try:
                for name, result in fit_all_baselines(X_train, y_train, X_test, y_test, self.config).items():
                    fold_results[name]['train'].append(result['train_metrics'])
                    fold_results[name]['test'].append(result['test_metrics'])
            except Exception as e:
                traceback.print_exc()

            print("\n  IA-BMLR")
            ia_result = None
            try:
                ia_result = fit_ia_bmlr_fold(X_train, y_train, X_test, y_test, self.config, verbose=True)
                fold_results['ia_bmlr']['train'].append(ia_result['train_metrics'])
                fold_results['ia_bmlr']['test'].append(ia_result['test_metrics'])
                learned_gammas.append(ia_result['learned_gamma'])
                last_fold_models['ia_bmlr'] = ia_result['model']
                if 'test_uncertainty' in ia_result:
                    unc_folds['ia_bmlr'].append(ia_result['test_uncertainty'])
            except Exception as e:
                traceback.print_exc()

            print("\n  STANDARD BMLR")
            std_result = None
            try:
                std_result = fit_standard_bmlr_fold(X_train, y_train, X_test, y_test, self.config, verbose=True)
                fold_results['standard_bmlr']['train'].append(std_result['train_metrics'])
                fold_results['standard_bmlr']['test'].append(std_result['test_metrics'])
                last_fold_models['standard_bmlr'] = std_result['model']
                if 'test_uncertainty' in std_result:
                    unc_folds['standard_bmlr'].append(std_result['test_uncertainty'])
            except Exception as e:
                traceback.print_exc()

            if self.config.get('KAPPA_SENSITIVITY', {}).get('run', True) \
                    and not kappa_sensitivity_done:
                try:
                    kappa_results = run_kappa_sensitivity(X_train, y_train, X_test, y_test, self.config)
                    dataset_results['kappa_sensitivity'] = kappa_results
                    kappa_path = os.path.join(dataset_results_dir, 'kappa_sensitivity.json')
                    save_results(kappa_results, kappa_path)
                except Exception as e:
                    traceback.print_exc()
                kappa_sensitivity_done = True

            fold_unc_path = os.path.join(
                dataset_results_dir, f'fold_{fold_num}_uncertainty.json')
            fold_unc_snapshot = {
                'fold':          fold_num,
                'ia_bmlr':       ia_result.get('test_uncertainty', {}) if ia_result else {},
                'standard_bmlr': std_result.get('test_uncertainty', {}) if std_result else {},
            }
            save_results(fold_unc_snapshot, fold_unc_path)

            if self.config.get('SINGLE_SPLIT', False):
                break

        method_names = {
            'multinomial_lr': 'Multinomial LR', 'random_forest': 'Random Forest',
            'svm': 'SVM (RBF)', 'xgboost': 'XGBoost',
            'ia_bmlr': 'IA-BMLR', 'standard_bmlr': 'Standard BMLR',
        }

        for model_name, fold_data in fold_results.items():
            if not fold_data['test']:
                continue
            agg_test  = aggregate_cv_results(fold_data['test'])
            agg_train = aggregate_cv_results(fold_data['train']) if fold_data['train'] else {}

            dataset_results['models'][model_name] = {
                'method':     method_names.get(model_name, model_name),
                'cv_metrics': agg_test,
                'test_metrics': {
                    k: agg_test.get(f'{k}_mean', np.nan)
                    for k in ['f1','accuracy','balanced_accuracy','precision', 'recall','roc_auc_ovr','log_loss','g_mean']
                },
                'test_metrics_std': {
                    k: agg_test.get(f'{k}_std', np.nan)
                    for k in ['f1','accuracy','balanced_accuracy','precision', 'recall','roc_auc_ovr','log_loss','g_mean']
                },
                'train_metrics': {
                    k: agg_train.get(f'{k}_mean', np.nan)
                    for k in ['f1','accuracy','balanced_accuracy','precision', 'recall','roc_auc_ovr','log_loss','g_mean']
                },
                'train_metrics_std': {
                    k: agg_train.get(f'{k}_std', np.nan)
                    for k in ['f1','accuracy','balanced_accuracy','precision', 'recall','roc_auc_ovr','log_loss','g_mean']
                },
                'n_folds': len(fold_data['test']),
            }

        for bm in ['ia_bmlr', 'standard_bmlr']:
            if unc_folds[bm] and bm in dataset_results['models']:
                dataset_results['models'][bm]['uncertainty_summary'] =  _aggregate_uncertainty_across_folds(unc_folds[bm])

        if learned_gammas and 'ia_bmlr' in dataset_results['models']:
            dataset_results['models']['ia_bmlr']['learned_gamma_mean'] = float(np.mean(learned_gammas))
            dataset_results['models']['ia_bmlr']['learned_gamma_std']  = float(np.std(learned_gammas))

        dataset_results['cv_results'] = {
            name: {'train': data['train'], 'test': data['test']}
            for name, data in fold_results.items() if data['test']
        }

        for bm_key, bm_prefix in [('ia_bmlr','ia_bmlr_last_fold'), ('standard_bmlr','standard_bmlr_last_fold')]:
            if bm_key in last_fold_models:
                try:
                    last_fold_models[bm_key]._save_results(dataset_results_dir, dataset_plots_dir, bm_prefix, verbose=True)
                except Exception as e:
                    print(f"  Warning: Could not save {bm_key} traces: {e}")

        print(f"\nCompleted: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        return dataset_results

    def run_all_datasets(self, dataset_names=None):
        if dataset_names is None:
            dataset_names = list(DATASETS.keys())
        for i, name in enumerate(dataset_names, 1):
            if name not in DATASETS:
                continue
            results = self.run_single_dataset(name, DATASETS[name])
            self.results[name] = results
            self.save_dataset_results(name, results)
            self.save_all_results()
            self.print_summary(name, results)
        return self.results

    def save_dataset_results(self, dataset_name, results):
        path = os.path.join(self.config['RESULTS_DIR'], dataset_name, 'dataset_results.json')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_results(results, path)

    def save_all_results(self):
        save_results(self.results, os.path.join(self.config['RESULTS_DIR'], 'all_results.json'))

    def print_summary(self, dataset_name, dataset_results):
        def safe(d, k, default=0.0):
            v = d.get(k, default)
            return default if (v is None or (isinstance(v, float) and np.isnan(v))) else v

        models = dataset_results.get('models', {})
        for model_name, res in sorted(
                models.items(),
                key=lambda x: safe(x[1].get('test_metrics', {}), 'f1'), reverse=True):
            if 'test_metrics' not in res:
                continue

        ia = models.get('ia_bmlr', {})
        if 'uncertainty_summary' in ia:
            u = ia['uncertainty_summary']
            for label, key in [
                ("Mean HDI width",        'mean_hdi_width_overall'),
                ("Mean pred entropy",     'mean_pred_entropy'),
                ("Fraction uncertain",    'frac_uncertain_predictions'),
            ]:
                if f'{key}_mean' in u:
                    print(f"    {label:<22}: "
                          f"{u[f'{key}_mean']:.4f} ± {u.get(f'{key}_std', 0):.4f}")

        if 'learned_gamma_mean' in ia:
            print(f"\n  Learned gamma: "
                  f"{ia['learned_gamma_mean']:.4f} ± {ia['learned_gamma_std']:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="IA-BMLR experiment runner. "
    )
    parser.add_argument(
        '--dataset', type=str, default=None,
        help="Name of a single dataset key from DATASETS"
    )
    return parser.parse_args()


def main():
    args   = parse_args()
    runner = ExperimentRunner()

    if args.dataset is not None:
        name = args.dataset
        if name not in DATASETS:
            sys.exit(1)
        results = runner.run_single_dataset(name, DATASETS[name])
        runner.save_dataset_results(name, results)
        runner.print_summary(name, results)
    else:
        runner.run_all_datasets()


if __name__ == "__main__":
    main()
