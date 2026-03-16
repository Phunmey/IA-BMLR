"""
Standard BMLR: Bayesian Multinomial Logistic Regression (Baseline)

Inherits from IABMLR but uses standard categorical likelihood (no weighting).
Used as baseline to compare against IA-BMLR (weighted likelihood).
"""

import os
import numpy as np
import pymc as pm
import arviz as az
import pytensor.tensor as pt
import matplotlib.pyplot as plt

from model_ia_bmlr import IABMLR

# Suppress OpenMP warning
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class StandardBMLR(IABMLR):
    """
    Standard Bayesian Multinomial Logistic Regression.
    
    Inherits from IABMLR but overrides fitting to use standard
    categorical likelihood (no weighting).
    """

    def __init__(self, prior_sigma=1.0,
                 n_samples=2000, n_tune=1000, n_chains=2, cores=None, target_accept=0.95):
        """
        Parameters
        ----------
        prior_sigma : float
            Prior standard deviation (same for all classes).
        n_samples, n_tune, n_chains, target_accept : MCMC settings.
        """
        # Initialize parent with gamma=0 (no entropy weighting needed)
        super().__init__(
            prior_sigma=prior_sigma,
            gamma_prior_sigma=1.0,  # Not used in standard BMLR
            les_neighbors=10,  # Not used in standard BMLR
            n_samples=n_samples,
            n_tune=n_tune,
            n_chains=n_chains,
            cores=cores,
            target_accept=target_accept
        )

    def fit(self, X, y, verbose=True, save_trace=True,
            data_dir="./outputs/results", plot_dir="./outputs/plots",
            filename_prefix="standard_bmlr"):
        """Fit the Standard BMLR model (no weighting)."""
        X = np.asarray(X)
        y = np.asarray(y)

        # Store references
        self._X_train = X
        self._y_train = y

        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        self.n_features = X.shape[1]
        N = len(y)

        # Map labels to contiguous indices
        self.label_to_idx = {c: i for i, c in enumerate(self.classes_)}
        self.idx_to_label = {i: c for c, i in self.label_to_idx.items()}
        y_idx = np.array([self.label_to_idx[yi] for yi in y])

        # Reference class (majority)
        counts = np.array([np.sum(y_idx == i) for i in range(self.n_classes)])
        self.ref_class = np.argmax(counts)

        if verbose:
            print("\n" + "=" * 60)
            print("Standard BMLR: Bayesian MLR (No Weighting)")
            print("=" * 60)
            print(f"\nDataset: N={N}, K={self.n_classes}, p={self.n_features}")
            print(f"Classes: {self.classes_}")
            print(f"Class counts: {dict(zip(self.classes_, counts))}")
            print(f"Reference class: {self.classes_[self.ref_class]} (index {self.ref_class})")
            print(f"\nFitting Standard BMLR (sigma={self.prior_sigma})...")

        self._fit_standard_bmlr(X, y_idx, verbose)

        if save_trace:
            os.makedirs(data_dir, exist_ok=True)
            os.makedirs(plot_dir, exist_ok=True)
            self._save_results(data_dir, plot_dir, filename_prefix, verbose)

        return self

    def _fit_standard_bmlr(self, X, y_idx, verbose=True):
        """Fit standard BMLR with categorical likelihood (NO weighting)."""
        coef_class_names = [f"class_{k}" for k in range(self.n_classes) if k != self.ref_class]
        coords = {
            'classes': coef_class_names,
            'features': [f"x{j}" for j in range(self.n_features)]
        }

        with pm.Model(coords=coords) as model:
            # Data containers
            X_data = pm.Data("X_data", X)
            y_data = pm.Data("y_data", y_idx)

            # Priors (K-1 classes)
            betaI = pm.Normal('betaI', mu=0, sigma=self.prior_sigma, dims='classes')
            beta = pm.Normal('beta', mu=0, sigma=self.prior_sigma, dims=('classes', 'features'))

            # Build full coefficient matrix
            betaI_full = pt.concatenate([betaI[:self.ref_class], pt.zeros(1), betaI[self.ref_class:]], axis=0)

            beta_full = pt.concatenate(
                [beta[:self.ref_class, :], pt.zeros((1, self.n_features)), beta[self.ref_class:, :]], axis=0)

            # Linear predictor
            logits = betaI_full + pm.math.dot(X_data, beta_full.T)

            # Probabilities
            proba = pm.Deterministic("proba", pt.special.softmax(logits, axis=1))

            # Standard categorical likelihood (NO weighting)
            pm.Categorical("y_obs", p=proba, observed=y_data)

            # Sample
            trace = pm.sample(
                draws=self.n_samples,
                tune=self.n_tune,
                chains=self.n_chains,
                cores=self.cores,
                target_accept=self.target_accept,
                return_inferencedata=True,
                progressbar=verbose,
                idata_kwargs={'log_likelihood': False}
            )

        self.model = model
        self.trace = trace

        proba_xr = trace.posterior['proba']  # (chain, draw, N, K)
        samples = proba_xr.stack(sample=('chain', 'draw')).values  # (N, K, M*C)
        samples = np.moveaxis(samples, -1, 0)  # (M*C, N, K)

        self._train_proba_samples = samples  # full array — enables HDI on training data
        self._train_proba = samples.mean(axis=0)  # posterior mean — same as before

        self.convergence_summary = az.summary(trace, var_names=['betaI', 'beta'], hdi_prob=0.95, round_to=3)

    def _save_results(self, data_dir, plot_dir, filename_prefix, verbose):
        """Save trace plots and summary (no LES for Standard BMLR)."""
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
        """Get summary of model configuration."""
        return {
            'prior_sigma': self.prior_sigma,
            'weighted_likelihood': False,
        }

