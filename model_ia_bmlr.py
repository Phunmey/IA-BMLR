"""
This code is for the manuscript on IA-BMLR: Imbalance-Aware Bayesian Multinomial Logistic Regression.
"""

import os
import warnings
import numpy as np
import pymc as pm
import arviz as az
import pytensor.tensor as pt
import matplotlib.pyplot as plt
from les_computation import compute_normalized_les
from weights_computation import compute_class_weights
from utils import compute_metrics

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class IABMLR:
    def __init__(self, prior_sigma=1.0, gamma_prior_sigma=1.0, les_neighbors=10, n_samples=2000, n_tune=1000,
                 n_chains=2, cores=None, target_accept=0.95):

        self.prior_sigma = prior_sigma
        self.gamma_prior_sigma = gamma_prior_sigma
        self.les_neighbors = les_neighbors
        self.n_samples = n_samples
        self.n_tune = n_tune
        self.n_chains = n_chains
        self.cores = cores
        self.target_accept = target_accept

        self.model = None
        self.trace = None
        self.n_classes = None
        self.n_features = None
        self.classes_ = None
        self.ref_class = None
        self.les_scores = None
        self.class_weights = None
        self.convergence_summary = None
        self.learned_gamma = None

        self.label_to_idx = None
        self.idx_to_label = None

        self._X_train = None
        self._y_train = None
        self._train_proba = None
        self._train_proba_samples = None

    def fit(self, X, y, verbose=True, save_trace=True, data_dir="./outputs/results", plot_dir="./outputs/plots",
            filename_prefix="ia_bmlr"):

        X = np.asarray(X)
        y = np.asarray(y)

        self._X_train = X
        self._y_train = y

        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        self.n_features = X.shape[1]
        N = len(y)

        self.label_to_idx = {c: i for i, c in enumerate(self.classes_)}
        self.idx_to_label = {i: c for c, i in self.label_to_idx.items()}
        y_idx = np.array([self.label_to_idx[yi] for yi in y])

        unique, counts = np.unique(y_idx, return_counts=True)
        self.ref_class = unique[np.argmax(counts)]

        self.les_scores, _, _ = compute_normalized_les(X, y, n_neighbors=self.les_neighbors)

        self.class_weights, class_weight_dict = compute_class_weights(y)
        C = np.eye(self.n_classes, dtype=np.float64)[y_idx]

        self._fit_weighted_bmlr(X, y_idx, C, N, verbose)
        if self.trace is not None:
            gamma_samples = self.trace.posterior['gamma'].values.flatten()
            self.learned_gamma = float(np.mean(gamma_samples))

        if save_trace:
            os.makedirs(data_dir, exist_ok=True)
            os.makedirs(plot_dir, exist_ok=True)
            self._save_results(data_dir, plot_dir, filename_prefix, verbose)

        return self

    def _fit_weighted_bmlr(self, X, y_idx, C, N_train, verbose=True):
        coef_class_names = [f"class_{k}" for k in range(self.n_classes) if k != self.ref_class]
        coords = {
            'classes': coef_class_names,
            'features': [f"x{j}" for j in range(self.n_features)]
        }

        with pm.Model(coords=coords) as model:
            X_data = pm.Data("X_data", X)
            y_data = pm.Data("y_data", y_idx)
            H_data = pm.Data("H_data", self.les_scores)
            C_data = pm.Data("C_data", C)

            gamma = pm.HalfNormal('gamma', sigma=self.gamma_prior_sigma)
            h = (1.0 + H_data) ** gamma
            S_k = pt.dot(C_data.T, h)
            S_obs = pt.dot(C_data, S_k)
            w = (N_train / self.n_classes) * (h / S_obs)
            weights_data = pm.Deterministic("weights_data", w)

            betaI = pm.Normal('betaI', mu=0, sigma=self.prior_sigma, dims='classes')
            beta = pm.Normal('beta', mu=0, sigma=self.prior_sigma, dims=('classes', 'features'))

            betaI_full = pt.concatenate([betaI[:self.ref_class], pt.zeros(1), betaI[self.ref_class:]], axis=0)
            beta_full = pt.concatenate(
                [beta[:self.ref_class, :], pt.zeros((1, self.n_features)), beta[self.ref_class:, :]], axis=0)

            logits = betaI_full + pm.math.dot(X_data, beta_full.T)
            proba = pm.Deterministic("proba", pt.special.softmax(logits, axis=1))

            log_idx = pt.arange(proba.shape[0])
            log_lik_per_obs = pt.log(proba[log_idx, y_data] + 1e-10)
            weighted_log_lik = pt.sum(weights_data * log_lik_per_obs)
            pm.Potential("weighted_likelihood", weighted_log_lik)

            trace = pm.sample(draws=self.n_samples, tune=self.n_tune, chains=self.n_chains, cores=self.cores,
                              target_accept=self.target_accept, return_inferencedata=True, progressbar=verbose,
                              idata_kwargs={'log_likelihood': False})

        self.model = model
        self.trace = trace

        proba_xr = trace.posterior['proba']
        samples = proba_xr.stack(sample=('chain', 'draw')).values
        samples = np.moveaxis(samples, -1, 0)

        self._train_proba_samples = samples
        self._train_proba = samples.mean(axis=0)
        self.convergence_summary = az.summary(trace, var_names=['gamma', 'betaI', 'beta'], hdi_prob=0.95, round_to=3)

    def _save_results(self, data_dir, plot_dir, filename_prefix, verbose):
        trace_plot = os.path.join(plot_dir, f"trace_{filename_prefix}.pdf")
        summary_file = os.path.join(data_dir, f"summary_{filename_prefix}.csv")
        les_file = os.path.join(data_dir, f"les_scores_{filename_prefix}.csv")

        az.plot_trace(self.trace, var_names=['gamma', 'betaI', 'beta'], compact=True)
        plt.tight_layout()
        plt.savefig(trace_plot, bbox_inches='tight')
        plt.close()

        self.convergence_summary.to_csv(summary_file)
        np.savetxt(les_file, self.les_scores, delimiter=',')

    def _get_posterior_samples(self, X, use_training_cache=False):
        if self.trace is None:
            raise ValueError("Model must be fitted before prediction.")

        X = np.asarray(X)

        if use_training_cache and self._train_proba_samples is not None:
            return self._train_proba_samples

        with self.model:
            pm.set_data({"X_data": X})
            ppc = pm.sample_posterior_predictive(self.trace, var_names=['proba'], random_seed=42, progressbar=True)

        proba_xr = ppc.posterior_predictive['proba']
        samples = proba_xr.stack(sample=('chain', 'draw')).values
        samples = np.moveaxis(samples, -1, 0)
        return samples

    def predict_proba(self, X, use_training_cache=False):
        samples = self._get_posterior_samples(X, use_training_cache)
        return samples.mean(axis=0)

    def _compute_hdi(self, samples, hdi_prob):
        MC, N, K = samples.shape

        samples_2d = samples.reshape(MC, N * K)
        hdi_flat = az.hdi(samples_2d, hdi_prob=hdi_prob, input_core_dims=[["draw"]])
        hdi_bounds = hdi_flat.reshape(N, K, 2)

        hdi_low = hdi_bounds[..., 0]
        hdi_high = hdi_bounds[..., 1]
        hdi_width = hdi_high - hdi_low

        return hdi_low, hdi_high, hdi_width

    def predict_uncertainty(self, X, hdi_prob=0.95, use_training_cache=False):
        samples = self._get_posterior_samples(X, use_training_cache)
        proba_mean = samples.mean(axis=0)
        hdi_low, hdi_high, hdi_width = self._compute_hdi(samples, hdi_prob)

        eps = 1e-12
        pred_entropy = -np.sum(proba_mean * np.log(proba_mean + eps), axis=1)

        y_pred_idx = np.argmax(proba_mean, axis=1)
        y_pred = np.array([self.idx_to_label[i] for i in y_pred_idx])

        return {'proba_mean': proba_mean, 'hdi_low': hdi_low, 'hdi_high': hdi_high, 'hdi_width': hdi_width,
                'pred_entropy': pred_entropy, 'y_pred': y_pred}

    def predict(self, X, use_training_cache=False):
        proba = self.predict_proba(X, use_training_cache)
        y_pred_idx = np.argmax(proba, axis=1)
        y_pred = np.array([self.idx_to_label[i] for i in y_pred_idx])
        return y_pred, proba

    def evaluate(self, X, y, use_training_cache=False, compute_uncertainty=False, hdi_prob=0.95):
        X = np.asarray(X)
        y = np.asarray(y)

        samples = self._get_posterior_samples(X, use_training_cache)
        proba_mean = samples.mean(axis=0)

        y_pred_idx = np.argmax(proba_mean, axis=1)
        y_pred = np.array([self.idx_to_label[i] for i in y_pred_idx])

        y_true_idx = np.array([self.label_to_idx[yi] for yi in y])
        y_pred_idx_mapped = np.array([self.label_to_idx[yi] for yi in y_pred])

        metrics = compute_metrics(y_true_idx, y_pred_idx_mapped, y_pred_proba=proba_mean)

        if compute_uncertainty:
            hdi_low, hdi_high, hdi_width = self._compute_hdi(samples, hdi_prob)
            eps = 1e-12
            pred_entropy = -np.sum(proba_mean * np.log(proba_mean + eps), axis=1)

            uncertainty = {'proba_mean': proba_mean, 'hdi_low': hdi_low, 'hdi_high': hdi_high, 'hdi_width': hdi_width,
                           'pred_entropy': pred_entropy, 'y_pred': y_pred, 'y_true': y}
            return metrics, uncertainty

        return metrics, None

    def evaluate_train_test(self, X_test, y_test, compute_uncertainty=False, hdi_prob=0.95):
        train_metrics, train_uncertainty = self.evaluate(self._X_train, self._y_train, use_training_cache=True,
                                                         compute_uncertainty=compute_uncertainty, hdi_prob=hdi_prob)
        test_metrics, test_uncertainty = self.evaluate(X_test, y_test, use_training_cache=False,
                                                       compute_uncertainty=compute_uncertainty, hdi_prob=hdi_prob)
        return train_metrics, test_metrics, train_uncertainty, test_uncertainty

    def get_summary(self):
        return {
            'prior_sigma': self.prior_sigma,
            'gamma_prior_sigma': self.gamma_prior_sigma,
            'learned_gamma': self.learned_gamma,
            'les_neighbors': self.les_neighbors,
            'les_min': float(self.les_scores.min()) if self.les_scores is not None else None,
            'les_max': float(self.les_scores.max()) if self.les_scores is not None else None,
            'les_mean': float(self.les_scores.mean()) if self.les_scores is not None else None,
        }
