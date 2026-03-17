"""
This code is for the Standard BMLR: Bayesian Multinomial Logistic Regression
"""

import os
import numpy as np
import pymc as pm
import arviz as az
import pytensor.tensor as pt
import matplotlib.pyplot as plt

from model_ia_bmlr import IABMLR

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class StandardBMLR(IABMLR):
    def __init__(self, prior_sigma=1.0,
                 n_samples=2000, n_tune=1000, n_chains=2, cores=None, target_accept=0.95):
        super().__init__(
            prior_sigma=prior_sigma,
            gamma_prior_sigma=1.0,
            les_neighbors=10,
            n_samples=n_samples,
            n_tune=n_tune,
            n_chains=n_chains,
            cores=cores,
            target_accept=target_accept
        )

    def fit(self, X, y, verbose=True, save_trace=True, data_dir="./outputs/results", plot_dir="./outputs/plots", filename_prefix="standard_bmlr"):
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

        counts = np.array([np.sum(y_idx == i) for i in range(self.n_classes)])
        self.ref_class = np.argmax(counts)
        self._fit_standard_bmlr(X, y_idx, verbose)

        if save_trace:
            os.makedirs(data_dir, exist_ok=True)
            os.makedirs(plot_dir, exist_ok=True)
            self._save_results(data_dir, plot_dir, filename_prefix, verbose)

        return self

    def _fit_standard_bmlr(self, X, y_idx, verbose=True):
        coef_class_names = [f"class_{k}" for k in range(self.n_classes) if k != self.ref_class]
        coords = {
            'classes': coef_class_names,
            'features': [f"x{j}" for j in range(self.n_features)]
        }

        with pm.Model(coords=coords) as model:
            X_data = pm.Data("X_data", X)
            y_data = pm.Data("y_data", y_idx)

            betaI = pm.Normal('betaI', mu=0, sigma=self.prior_sigma, dims='classes')
            beta = pm.Normal('beta', mu=0, sigma=self.prior_sigma, dims=('classes', 'features'))

            betaI_full = pt.concatenate([betaI[:self.ref_class], pt.zeros(1), betaI[self.ref_class:]], axis=0)

            beta_full = pt.concatenate(
                [beta[:self.ref_class, :], pt.zeros((1, self.n_features)), beta[self.ref_class:, :]], axis=0)

            logits = betaI_full + pm.math.dot(X_data, beta_full.T)

            proba = pm.Deterministic("proba", pt.special.softmax(logits, axis=1))

            pm.Categorical("y_obs", p=proba, observed=y_data)

            trace = pm.sample(draws=self.n_samples, tune=self.n_tune, chains=self.n_chains, cores=self.cores, target_accept=self.target_accept, return_inferencedata=True, progressbar=verbose, idata_kwargs={'log_likelihood': False})

        self.model = model
        self.trace = trace

        proba_xr = trace.posterior['proba']
        samples = proba_xr.stack(sample=('chain', 'draw')).values
        samples = np.moveaxis(samples, -1, 0)

        self._train_proba_samples = samples
        self._train_proba = samples.mean(axis=0)

        self.convergence_summary = az.summary(trace, var_names=['betaI', 'beta'], hdi_prob=0.95, round_to=3)

    def _save_results(self, data_dir, plot_dir, filename_prefix, verbose):
        trace_plot = os.path.join(plot_dir, f"trace_{filename_prefix}.pdf")
        summary_file = os.path.join(data_dir, f"summary_{filename_prefix}.csv")

        az.plot_trace(self.trace, var_names=['betaI', 'beta'], compact=True)
        plt.tight_layout()
        plt.savefig(trace_plot, bbox_inches='tight')
        plt.close()

        self.convergence_summary.to_csv(summary_file)

        if verbose:
            print(f"  Saved trace plot: {trace_plot}")
            print(f"  Saved summary: {summary_file}")

    def get_summary(self):
        return {
            'prior_sigma': self.prior_sigma,
            'weighted_likelihood': False,
        }

