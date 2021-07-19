import numpy as onp
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvnpy
from autograd.scipy.stats import multivariate_normal as mvn
import autograd.numpy as np
from autograd import grad, value_and_grad
from autograd.misc.optimizers import adam
from gp_functions import rbf_covariance
from scipy.optimize import minimize
from os.path import join as pjoin

import matplotlib

font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

SAVE_DIR = "/path/to/save/dir"


class LMC:
    def __init__(
        self,
        X,
        Y,
        kernel,
        n_latent_dims=2,
        n_spatial_dims=2,
        n_kernel_params=2,
        n_noise_variance_params=1,
    ):

        if X.shape[0] != Y.shape[0]:
            raise ValueError("Number of samples do not match between X and Y.")

        self.X = X
        self.Y = Y
        self.kernel = kernel
        self.n_latent_dims = n_latent_dims
        self.n_spatial_dims = n_spatial_dims
        self.n_kernel_params = n_kernel_params
        self.n_noise_variance_params = n_noise_variance_params

        self.N = X.shape[0]
        self.n_features = Y.shape[1]

    def unpack_params(self, params, n_kernel_params):
        noise_variance = np.exp(params[0]) + 0.001
        kernel_params = params[1 : n_kernel_params + 1]
        W = np.reshape(
            params[n_kernel_params + 1 :], (self.n_latent_dims, self.n_features)
        )
        return W, noise_variance, kernel_params

    def gp_likelihood(self, params):
        W, noise_variance, kernel_params = self.unpack_params(
            params, self.n_kernel_params
        )

        # Compute log likelihood
        cov_xx = self.kernel(X, X, kernel_params) + noise_variance * np.eye(self.N)
        cov = np.kron(cov_xx, W.T @ W) + 0.01 * np.eye(self.N * self.n_features)
        LL = mvn.logpdf(self.Y.flatten(), np.zeros(self.N * self.n_features), cov)

        return -LL

    def summary(self, pars):
        print("LL {0:1.3e}".format(-self.gp_likelihood(pars)))

    def fit(self, plot_updates=False):
        param_init = np.concatenate(
            [
                np.random.normal(size=self.n_noise_variance_params),  # Noise variance
                np.random.normal(size=self.n_kernel_params),  # GP params
                np.random.normal(
                    scale=1, size=self.n_latent_dims * self.n_features
                ),  # W (loadings)
            ]
        )

        res = minimize(
            value_and_grad(self.gp_likelihood),
            param_init,
            jac=True,
            method="CG",
            callback=self.summary,
        )
        W, noise_variance, kernel_params = self.unpack_params(
            res.x, self.n_kernel_params
        )
        return W, noise_variance, kernel_params


if __name__ == "__main__":

    n_features = 2
    n_latent_dims = 1
    kernel = rbf_covariance
    kernel_params_true = np.array([1, 1.0])
    n = 50
    ntest = 50
    sigma2 = 0.1
    n_spatial_dims = 1
    X_full = np.vstack(
        [np.linspace(-7, 7, n + ntest, 1) for _ in range(n_spatial_dims)]
    ).T
    W_orig = np.array([[-2.0, 2.0]])
    F_orig = np.vstack(
        [
            mvnpy.rvs(
                mean=np.zeros(n + ntest), cov=kernel(X_full, X_full, kernel_params_true)
            )
            for _ in range(n_latent_dims)
        ]
    ).T
    Y_full = F_orig @ W_orig + np.random.normal(
        scale=np.sqrt(sigma2), size=(n + ntest, n_features)
    )

    X = X_full[:n]
    Y = Y_full[:n]
    Xtest = X_full[n:]
    Ytest = Y_full[n:]

    warp_gp = LMC(
        X,
        Y,
        kernel=rbf_covariance,
        n_latent_dims=n_latent_dims,
        n_spatial_dims=n_spatial_dims,
    )
    W_fitted, noise_variance, kernel_params = warp_gp.fit(plot_updates=False)

    ## Make predictions
    nnew = 75
    xnew_lim = 12
    xnew = np.linspace(-xnew_lim, xnew_lim, nnew)
    Xnew = np.expand_dims(xnew, 1)

    WWT = np.outer(W_fitted.T, W_fitted)
    Kxx = np.kron(rbf_covariance(X, X, kernel_params), WWT)
    Kxx += noise_variance * np.eye(Kxx.shape[0])

    Kxnewxnew = np.kron(rbf_covariance(Xnew, Xnew, kernel_params), WWT)
    Kxxnew = np.kron(rbf_covariance(X, Xnew, kernel_params), WWT)
    Kxx_inv = np.linalg.solve(Kxx, np.eye(Kxx.shape[0]))

    # Y_flattened
    mean_pred = Kxxnew.T @ Kxx_inv @ np.ndarray.flatten(Y, "C")
    mean_pred = np.reshape(mean_pred, (nnew, n_features))

    Xaugmented = np.concatenate([X, Xtest], axis=0)
    Kxx_augmented = rbf_covariance(Xaugmented, Xaugmented, kernel_params)

    Kxx_augmented_full = np.kron(WWT, Kxx_augmented)
    Kxx = Kxx_augmented_full[
        : n * n_features + ntest, : n * n_features + ntest
    ] + 0.01 * np.eye(n * n_features + ntest)

    Kxxtest = Kxx_augmented_full[: n * n_features + ntest, n * n_features + ntest :]
    Kxx_inv = np.linalg.solve(Kxx, np.eye(n * n_features + ntest))

    Y_for_preds = np.concatenate([Y[:, 0], Ytest[:, 0], Y[:, 1]])
    preds = Kxxtest.T @ Kxx_inv @ Y_for_preds

    ## Get normal GP predictions for the features independently
    Kxx = rbf_covariance(X, X, kernel_params)
    Kxx += noise_variance * np.eye(Kxx.shape[0])
    Kxx_inv = np.linalg.solve(Kxx, np.eye(Kxx.shape[0]))
    Kxxnew = rbf_covariance(X, Xnew, kernel_params)
    mean_pred_gp_y1 = Kxxnew.T @ Kxx_inv @ Y[:, 0]
    mean_pred_gp_y2 = Kxxnew.T @ Kxx_inv @ Y[:, 1]

    plt.figure(figsize=(10, 4))

    plt.scatter(X[:, 0], Y[:, 0], color="red", alpha=0.5, label="Y1")
    plt.scatter(X[:, 0], Y[:, 1], color="green", alpha=0.5, label="Y2")
    plt.plot(
        xnew,
        mean_pred_gp_y2,
        color="gray",
        linestyle="--",
        label="GP predictions",
        linewidth=5,
    )

    plt.plot(Xtest, preds, label="LMC predictions", color="black", linewidth=5)
    plt.scatter(Xtest, Ytest[:, 0], color="red", alpha=0.5)

    plt.xlabel(r"$X$")
    plt.ylabel(r"$Y$")
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(pjoin(SAVE_DIR, "lmc_2d_preds.png"))
    # plt.show()
    plt.close()

    plt.figure(figsize=(10, 4))

    plt.scatter(X[:, 0], Y[:, 0], color="red", alpha=0.5, label="Y1")
    plt.scatter(X[:, 0], Y[:, 1], color="green", alpha=0.5, label="Y2")

    plt.scatter(Xtest, Ytest[:, 0], color="red", alpha=0.5)
    plt.scatter(Xtest, Ytest[:, 1], color="green", alpha=0.5)

    plt.xlabel(r"$X$")
    plt.ylabel(r"$Y$")
    plt.legend(fontsize=10)
    plt.xlim([-xnew_lim, xnew_lim])
    plt.tight_layout()
    plt.savefig(pjoin(SAVE_DIR, "lmc_full_data.png"))
    # plt.show()
    plt.close()

    plt.figure(figsize=(10, 4))

    plt.scatter(X[:, 0], Y[:, 0], color="red", alpha=0.5, label="Y1")
    plt.scatter(X[:, 0], Y[:, 1], color="green", alpha=0.5, label="Y2")

    plt.scatter(Xtest, Ytest[:, 0], color="red", alpha=0.5)

    plt.xlabel(r"$X$")
    plt.ylabel(r"$Y$")
    plt.legend(fontsize=10)
    plt.xlim([-xnew_lim, xnew_lim])
    plt.tight_layout()
    plt.savefig(pjoin(SAVE_DIR, "lmc_missing_data.png"))
    # plt.show()
    plt.close()

    plt.figure(figsize=(10, 4))

    plt.scatter(X[:, 0], Y[:, 0], color="red", alpha=0.5, label="Y1")
    plt.scatter(X[:, 0], Y[:, 1], color="green", alpha=0.5, label="Y2")

    plt.scatter(Xtest, Ytest[:, 0], color="red", alpha=0.5)

    plt.plot(
        xnew,
        mean_pred_gp_y2,
        color="gray",
        linestyle="--",
        label="GP predictions",
        linewidth=5,
    )

    plt.xlabel(r"$X$")
    plt.ylabel(r"$Y$")
    plt.legend(fontsize=10)
    plt.xlim([-xnew_lim, xnew_lim])
    plt.tight_layout()
    plt.savefig(pjoin(SAVE_DIR, "lmc_partial_preds.png"))
    # plt.show()
    plt.close()

    import ipdb

    ipdb.set_trace()
