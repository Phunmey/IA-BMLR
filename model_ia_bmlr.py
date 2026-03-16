"""
This code is for the manuscript on IA-BMLR: Imbalance-Aware Bayesian Multinomial Logistic Regression.

- Local Entropy Score (LES) for identifying hard examples, normalized to [0, 1]
- Within-class normalized entropy weights that guarantee sum_i w_i = N
- Uniform priors on regression coefficients; HalfNormal prior on gamma
- Weighted log-likelihood via pm.Potential (not a standard likelihood)
w_i = (N / K) * (1 + H_i)^gamma / S_{y_i}(gamma); where S_k(gamma) = sum_{j: y_j=k} (1 + H_j)^gamma

This replaces the earlier unnormalized form w_i = w_class(y_i) * (1+H_i)^gamma.
The within-class normalization ensures the total information budget equals N for
any value of gamma, while preserving the clean conceptual separation between
between-class balancing (class weights) and within-class emphasis (entropy weights).
Because S_k depends on gamma, weights are computed inside the PyMC model as
differentiable PyTensor expressions.
"""

import os
import warnings
import numpy as np
import pymc as pm
import arviz as az
import pytensor.tensor as pt
import matplotlib.pyplot as plt
from les_computation import compute_normalized_les  # normalized H_i in [0,1]
from weights_computation import compute_class_weights  # class weights still precomputed outside model
from utils import compute_metrics

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class IABMLR:
    """
    Imbalance-Aware Bayesian Multinomial Logistic Regression.
    """

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
        self._train_proba = None  # posterior mean proba, shape (N_train, K)
        self._train_proba_samples = None  # full samples array, shape (M*C, N_train, K)

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

        # Step 1: Compute normalized Local Entropy Score (LES), H_i in [0, 1].
        self.les_scores, _, _ = compute_normalized_les(X, y, n_neighbors=self.les_neighbors)

        if verbose:
            print(
                f" LES stats: min={self.les_scores.min():.4f}, max={self.les_scores.max():.4f}, mean={self.les_scores.mean():.4f}")

        # Step 2: Compute class weights
        self.class_weights, class_weight_dict = compute_class_weights(y)
        C = np.eye(self.n_classes, dtype=np.float64)[y_idx]

        # Step 3: Build and fit weighted BMLR
        if verbose:
            print(f"\nStep 3: Fitting weighted BMLR (sigma={self.prior_sigma})...")

        self._fit_weighted_bmlr(X, y_idx, C, N, verbose)
        if self.trace is not None:
            gamma_samples = self.trace.posterior['gamma'].values.flatten()
            self.learned_gamma = float(np.mean(gamma_samples))

        # Step 4: Save results
        if save_trace:
            os.makedirs(data_dir, exist_ok=True)
            os.makedirs(plot_dir, exist_ok=True)
            self._save_results(data_dir, plot_dir, filename_prefix, verbose)

        return self

    def _fit_weighted_bmlr(self, X, y_idx, C, N_train, verbose=True):
        """
        Fit the weighted Bayesian MLR with within-class normalized entropy weights.

        Weight formula (derived in Section 2.2.4 of the paper):

            w_i = (N / K) * (1 + H_i)^gamma / S_{y_i}(gamma)

        where S_k(gamma) = sum_{j: y_j=k} (1 + H_j)^gamma is the within-class
        sum of raw entropy terms.  This ensures sum_i w_i = N exactly for any
        value of gamma, addressing Reviewer A's normalization concern.

        Because S_k depends on gamma (a PyMC random variable), the entire weight
        computation must live inside the model as differentiable PyTensor operations
        so that NUTS can propagate gradients through it automatically.
        """
        coef_class_names = [f"class_{k}" for k in range(self.n_classes) if k != self.ref_class]
        coords = {
            'classes': coef_class_names,
            'features': [f"x{j}" for j in range(self.n_features)]
        }

        with pm.Model(coords=coords) as model:
            X_data = pm.Data("X_data", X)
            y_data = pm.Data("y_data", y_idx)

            # Normalized LES scores H_i in [0, 1] — fixed constants, passed as Data
            # so PyMC tracks them in the computation graph without sampling them.
            H_data = pm.Data("H_data", self.les_scores)

            # One-hot class indicator matrix C, shape (N, K) — fixed constant.
            # Enables vectorized within-class summation via matrix products.
            C_data = pm.Data("C_data", C)

            # ── Prior on entropy focus parameter ──────────────────────────────
            gamma = pm.HalfNormal('gamma', sigma=self.gamma_prior_sigma)

            # ── Within-class normalized weight computation ────────────────────
            # Step 1: raw entropy terms for all N observations, shape (N,).
            # When gamma=0, h=1 everywhere and weights reduce to pure class weights.
            h = (1.0 + H_data) ** gamma

            # Step 2: within-class sums S_k, shape (K,).
            # C_data.T has shape (K, N), so this matrix product gives one sum per class:
            #   S_k = sum_{j: y_j=k} (1 + H_j)^gamma
            S_k = pt.dot(C_data.T, h)

            # Step 3: map S_{y_i} back to each observation, shape (N,).
            # C_data has shape (N, K), so C_data @ S_k assigns S_{y_i} to each row.
            S_obs = pt.dot(C_data, S_k)

            # Step 4: final normalized weights, shape (N,).
            # The n_{y_i} factor cancels algebraically (see Section 2.2.4),
            # leaving the simplified form w_i = (N/K) * h_i / S_{y_i}.
            # This guarantees sum_i w_i = N for any value of gamma.
            w = (N_train / self.n_classes) * (h / S_obs)
            weights_data = pm.Deterministic("weights_data", w)

            # Priors (K-1 classes, reference class has zeros)
            betaI = pm.Normal('betaI', mu=0, sigma=self.prior_sigma, dims='classes')
            beta = pm.Normal('beta', mu=0, sigma=self.prior_sigma, dims=('classes', 'features'))

            betaI_full = pt.concatenate([betaI[:self.ref_class], pt.zeros(1), betaI[self.ref_class:]], axis=0)
            beta_full = pt.concatenate(
                [beta[:self.ref_class, :], pt.zeros((1, self.n_features)), beta[self.ref_class:, :]], axis=0)

            logits = betaI_full + pm.math.dot(X_data, beta_full.T)
            proba = pm.Deterministic("proba", pt.special.softmax(logits, axis=1))

            # Weighted log-likelihood using Potential
            log_idx = pt.arange(proba.shape[0])
            log_lik_per_obs = pt.log(proba[log_idx, y_data] + 1e-10)
            weighted_log_lik = pt.sum(weights_data * log_lik_per_obs)
            pm.Potential("weighted_likelihood", weighted_log_lik)

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

        # Cache both the posterior mean and the full samples array for training data.
        # The full samples array has shape (M*C, N_train, K) — chains and draws are
        # stacked along axis 0. This is needed for computing training-set HDIs without
        # re-running sample_posterior_predictive.
        proba_xr = trace.posterior['proba']  # (chain, draw, N, K)
        samples = proba_xr.stack(sample=('chain', 'draw')).values  # (N, K, M*C)
        samples = np.moveaxis(samples, -1, 0)  # (M*C, N, K)

        self._train_proba_samples = samples
        self._train_proba = samples.mean(axis=0)  # (N, K)
        self.convergence_summary = az.summary(trace, var_names=['gamma', 'betaI', 'beta'], hdi_prob=0.95, round_to=3)

    def _save_results(self, data_dir, plot_dir, filename_prefix, verbose):
        """Save trace plots and summary."""
        trace_plot = os.path.join(plot_dir, f"trace_{filename_prefix}.pdf")
        summary_file = os.path.join(data_dir, f"summary_{filename_prefix}.csv")
        les_file = os.path.join(data_dir, f"les_scores_{filename_prefix}.csv")

        # Trace plot
        az.plot_trace(self.trace, var_names=['gamma', 'betaI', 'beta'], compact=True)
        plt.tight_layout()
        plt.savefig(trace_plot, bbox_inches='tight')
        plt.close()

        self.convergence_summary.to_csv(summary_file)
        np.savetxt(les_file, self.les_scores, delimiter=',')

    # ──────────────────────────────────────────────────────────────────────────
    # Core posterior sample retrieval
    # ──────────────────────────────────────────────────────────────────────────

    def _get_posterior_samples(self, X, use_training_cache=False):
        """
        Return the full posterior predictive sample array for X.

        This is the single entry point for all downstream prediction and
        uncertainty methods.  Running sample_posterior_predictive is the
        expensive step; everything else (posterior mean, HDIs, entropy) is
        cheap post-processing of this array.

        Parameters
        ----------
        X : ndarray, shape (N, p)
        use_training_cache : bool
            If True and X is the training set, return the cached sample array
            computed during fit() without re-running MCMC forward passes.

        Returns
        -------
        samples : ndarray, shape (M*C, N, K)
            Posterior predictive probabilities.  Axis 0 indexes MCMC samples
            (chains stacked), axis 1 indexes observations, axis 2 indexes classes.
        """
        if self.trace is None:
            raise ValueError("Model must be fitted before prediction.")

        X = np.asarray(X)

        # Return cached training samples if available, avoiding a redundant
        # sample_posterior_predictive call for the training set.
        if use_training_cache and self._train_proba_samples is not None:
            return self._train_proba_samples

        # For any other data (test set, new observations), generate posterior
        # predictive samples by pushing each MCMC draw through the softmax.
        with self.model:
            pm.set_data({"X_data": X})
            ppc = pm.sample_posterior_predictive(self.trace, var_names=['proba'], random_seed=42, progressbar=True)

        # ppc.posterior_predictive['proba'] has dims (chain, draw, N, K).
        # Stack chain and draw into a single sample axis → (M*C, N, K).
        proba_xr = ppc.posterior_predictive['proba']
        samples = proba_xr.stack(sample=('chain', 'draw')).values  # (N, K, M*C)
        samples = np.moveaxis(samples, -1, 0)  # (M*C, N, K)
        return samples

    # ──────────────────────────────────────────────────────────────────────────
    # Public prediction interface
    # ──────────────────────────────────────────────────────────────────────────

    def predict_proba(self, X, use_training_cache=False):
        """
        Return the posterior mean class probability for each observation.

        This is the point estimate used for classification (argmax) and for
        standard metrics (AUC, log-loss).  For a full uncertainty-aware summary
        including HDI bounds, use predict_uncertainty() instead.

        Returns
        -------
        proba : ndarray, shape (N, K)
            Posterior mean P(y = k | x_i, X, Y) for each observation i and class k.
        """
        samples = self._get_posterior_samples(X, use_training_cache)
        return samples.mean(axis=0)  # average over M*C samples → (N, K)

    def _compute_hdi(self, samples, hdi_prob):
        """
        Compute HDI bounds for a posterior samples array robustly across
        ArviZ versions.

        The issue: az.hdi behaviour on 3-dimensional numpy arrays varies across
        ArviZ versions.  Some versions return shape (N, K, 2) as expected; others
        flatten the non-sample dimensions first, returning (N*K, 2), which makes
        hdi_width 1-dimensional and breaks downstream indexing.

        The fix: always reshape the samples to 2D (M*C, N*K) before calling
        az.hdi — every ArviZ version handles 2D arrays consistently — then
        reshape the result back to (N, K, 2).

        Parameters
        ----------
        samples : ndarray, shape (M*C, N, K)
        hdi_prob : float

        Returns
        -------
        hdi_low  : ndarray, shape (N, K)
        hdi_high : ndarray, shape (N, K)
        hdi_width: ndarray, shape (N, K)
        """
        MC, N, K = samples.shape

        # Flatten N and K into a single axis so az.hdi always receives 2D input.
        samples_2d = samples.reshape(MC, N * K)  # (M*C, N*K)
        hdi_flat = az.hdi(samples_2d, hdi_prob=hdi_prob, input_core_dims=[["draw"]])  # (N*K, 2)
        hdi_bounds = hdi_flat.reshape(N, K, 2)  # (N, K, 2)

        hdi_low = hdi_bounds[..., 0]  # (N, K)
        hdi_high = hdi_bounds[..., 1]  # (N, K)
        hdi_width = hdi_high - hdi_low  # (N, K)

        return hdi_low, hdi_high, hdi_width


    def predict_uncertainty(self, X, hdi_prob=0.95, use_training_cache=False):
        """
        Return a full uncertainty summary for each observation and class.

        This exposes the distinctive advantage of the Bayesian framework: rather
        than a single probability estimate per class, we characterize the entire
        posterior predictive distribution via its mean and HDI bounds.

        A narrow HDI indicates the model is confident in its probability estimate
        for that class; a wide HDI indicates genuine uncertainty.  When the HDI
        bounds of the two most probable classes overlap for a given observation,
        the model is signalling that it cannot confidently distinguish between
        those classes — information that a point prediction alone cannot convey.

        Parameters
        ----------
        X : ndarray, shape (N, p)
        hdi_prob : float
            Probability mass enclosed by the HDI (default 0.95).
        use_training_cache : bool

        Returns
        -------
        uncertainty : dict with the following keys, each an ndarray:

            proba_mean   : shape (N, K)  — posterior mean P(y=k | x_i)
            hdi_low      : shape (N, K)  — lower HDI bound per observation per class
            hdi_high     : shape (N, K)  — upper HDI bound per observation per class
            hdi_width    : shape (N, K)  — HDI width (hdi_high - hdi_low); scalar
                                           measure of uncertainty per obs per class
            pred_entropy : shape (N,)    — Shannon entropy of the posterior mean
                                           probability vector; scalar summary of
                                           overall predictive uncertainty per obs
            y_pred       : shape (N,)    — predicted class label (argmax of proba_mean)
        """
        samples = self._get_posterior_samples(X, use_training_cache)
        # samples has shape (M*C, N, K)

        # Posterior mean: average over the sample axis
        proba_mean = samples.mean(axis=0)  # (N, K)

        # HDI bounds — use helper to ensure correct (N, K, 2) shape across
        # all ArviZ versions (see _compute_hdi docstring for details).
        hdi_low, hdi_high, hdi_width = self._compute_hdi(samples, hdi_prob)

        # Predictive entropy of the posterior mean vector: summarizes total
        # uncertainty about class membership in a single scalar per observation.
        # Entropy is high when probability mass is spread across classes (uncertain)
        # and low when mass is concentrated on one class (confident).
        eps = 1e-12
        pred_entropy = -np.sum(proba_mean * np.log(proba_mean + eps), axis=1)  # (N,)

        # Point prediction: argmax of posterior mean
        y_pred_idx = np.argmax(proba_mean, axis=1)
        y_pred = np.array([self.idx_to_label[i] for i in y_pred_idx])

        return {
            'proba_mean': proba_mean,
            'hdi_low': hdi_low,
            'hdi_high': hdi_high,
            'hdi_width': hdi_width,
            'pred_entropy': pred_entropy,
            'y_pred': y_pred,
        }

    def predict(self, X, use_training_cache=False):
        """
        Predict class labels and return posterior mean probabilities.

        Returns
        -------
        y_pred : ndarray, shape (N,)   — predicted class labels
        proba  : ndarray, shape (N, K) — posterior mean class probabilities
        """
        proba = self.predict_proba(X, use_training_cache)
        y_pred_idx = np.argmax(proba, axis=1)
        y_pred = np.array([self.idx_to_label[i] for i in y_pred_idx])
        return y_pred, proba

    # ──────────────────────────────────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate(self, X, y, use_training_cache=False, compute_uncertainty=False,
                 hdi_prob=0.95):
        """
        Compute performance metrics, optionally including uncertainty summaries.

        Parameters
        ----------
        X : ndarray, shape (N, p)
        y : array-like, shape (N,)   — true class labels
        use_training_cache : bool
        compute_uncertainty : bool
            If True, also compute and return the full uncertainty summary from
            predict_uncertainty().  This adds negligible cost because
            _get_posterior_samples() is called only once internally.
        hdi_prob : float
            HDI probability mass (only used when compute_uncertainty=True).

        Returns
        -------
        metrics : dict  — performance metrics from compute_metrics()
        uncertainty : dict or None
            Observation-level uncertainty summary (only when compute_uncertainty=True).
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # Retrieve posterior samples once; derive everything else from them.
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

            uncertainty = {
                'proba_mean': proba_mean,
                'hdi_low': hdi_low,
                'hdi_high': hdi_high,
                'hdi_width': hdi_width,
                'pred_entropy': pred_entropy,
                'y_pred': y_pred,
                'y_true': y,
            }
            return metrics, uncertainty

        return metrics, None

    def evaluate_train_test(self, X_test, y_test, compute_uncertainty=False, hdi_prob=0.95):
        """
        Evaluate on both training and test sets in one call.

        Returns
        -------
        train_metrics : dict
        test_metrics  : dict
        train_uncertainty : dict or None
        test_uncertainty  : dict or None
        """
        train_metrics, train_uncertainty = self.evaluate(self._X_train, self._y_train, use_training_cache=True,
                                                         compute_uncertainty=compute_uncertainty, hdi_prob=hdi_prob)
        test_metrics, test_uncertainty = self.evaluate(X_test, y_test, use_training_cache=False,
                                                       compute_uncertainty=compute_uncertainty, hdi_prob=hdi_prob)
        return train_metrics, test_metrics, train_uncertainty, test_uncertainty

    def get_summary(self):
        """Get summary of model configuration and LES statistics."""
        return {
            'prior_sigma': self.prior_sigma,
            'gamma_prior_sigma': self.gamma_prior_sigma,
            'learned_gamma': self.learned_gamma,
            'les_neighbors': self.les_neighbors,
            'les_min': float(self.les_scores.min()) if self.les_scores is not None else None,
            'les_max': float(self.les_scores.max()) if self.les_scores is not None else None,
            'les_mean': float(self.les_scores.mean()) if self.les_scores is not None else None,
        }
