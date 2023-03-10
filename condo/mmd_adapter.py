from copy import deepcopy
from typing import Union

import math
import numpy as np
import pandas as pd
import torch
import torchmin as tm
import sklearn.utils as skut

from sklearn.cluster import KMeans
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
from sklearn.covariance import GraphicalLassoCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    RBF,
    WhiteKernel,
)
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import OneHotEncoder

from condo.heteroscedastic_kernel import HeteroscedasticKernel
from condo.cat_kernels import (
    CatKernel,
    HeteroscedasticCatKernel,
)


def run_mmd(
    S: np.ndarray,
    T: np.ndarray,
    transform_type: str,
    debug: bool,
    verbose: Union[bool, int],
    epochs: int = 100,
    batch_size: int = 16,
    alpha: float = 1e-2,
    beta: float = 0.9,
):
    """
    Args:
        debug: is ignored

        epochs: number of times to pass through all observed confounder values.

        batch_size: number of samples to draw from S and T per confounder value.
            kernel matrix will be of size (batch_size, batch_size).
            The number of batches per epoch is num_S*num_T / (batch_size ** 2).

        alpha: gradient descent learning rate (ie step size).

        beta: Nesterov momentum

    Returns:
        M: (num_feats, num_feats)
        b: (num_feats,)
        debug_tuple: (m_plot, b_plot, mb_objs) or None
    """
    rng = np.random.RandomState(42)
    num_S = S.shape[0]
    num_T = T.shape[0]
    num_feats = S.shape[1]
    T_torch = torch.from_numpy(T)
    S_torch = torch.from_numpy(S)

    if transform_type == "location-scale":
        M = torch.ones(num_feats, dtype=torch.float64, requires_grad=True)
        b = torch.zeros(num_feats, dtype=torch.float64, requires_grad=True)
    elif transform_type == "affine":
        M = torch.eye(num_feats, num_feats, dtype=torch.float64, requires_grad=True)
        b = torch.zeros(num_feats, dtype=torch.float64, requires_grad=True)
    batches = math.ceil(num_S * num_T / (batch_size * batch_size))
    full_epochs = math.floor(epochs)
    frac_epochs = epochs % 1
    terms_per_batch = batch_size * batch_size
    obj_history = []
    best_M = np.eye(num_feats, num_feats)
    best_b = np.zeros(num_feats)
    for epoch in range(full_epochs + 1):
        epoch_start_M = M.detach().numpy()
        epoch_start_b = b.detach().numpy()
        Mz = torch.zeros_like(M)
        bz = torch.zeros_like(b)
        objs = np.zeros(batches)
        if epoch == full_epochs:
            cur_batches = round(frac_epochs * batches)
        else:
            cur_batches = batches
        for batch in range(cur_batches):
            tgtsample_ixs = rng.choice(num_T, size=batch_size, replace=True).tolist()
            srcsample_ixs = rng.choice(num_S, size=batch_size, replace=True).tolist()
            obj = torch.tensor(0.0, requires_grad=True)
            Tsample = T_torch[tgtsample_ixs, :]
            if transform_type == "location-scale":
                adaptedSsample = S_torch[srcsample_ixs, :] * M.reshape(
                    1, -1
                ) + b.reshape(1, -1)
            elif transform_type == "affine":
                adaptedSsample = S_torch[srcsample_ixs, :] @ M.T + b.reshape(1, -1)
            length_scale_np = (
                torch.mean((Tsample - adaptedSsample) ** 2, axis=0).detach().numpy()
            )
            invroot_length_scale = 1.0 / np.sqrt(length_scale_np)
            lscaler = torch.from_numpy(invroot_length_scale).view(1, -1)
            scaled_Tsample = Tsample * lscaler
            scaled_adaptedSsample = adaptedSsample * lscaler

            factor = 1.0 / terms_per_batch
            obj = obj - 2 * factor * torch.sum(
                torch.exp(
                    -1.0
                    / 2.0
                    * (
                        (scaled_Tsample @ scaled_Tsample.T).diag().unsqueeze(1)
                        - 2 * scaled_Tsample @ scaled_adaptedSsample.T
                        + (scaled_adaptedSsample @ scaled_adaptedSsample.T)
                        .diag()
                        .unsqueeze(0)
                    )
                )
            )
            obj = obj + factor * torch.sum(
                torch.exp(
                    -1.0
                    / 2.0
                    * (
                        (scaled_adaptedSsample @ scaled_adaptedSsample.T)
                        .diag()
                        .unsqueeze(1)
                        - 2 * scaled_adaptedSsample @ scaled_adaptedSsample.T
                        + (scaled_adaptedSsample @ scaled_adaptedSsample.T)
                        .diag()
                        .unsqueeze(0)
                    )
                )
            )
            obj.backward()
            with torch.no_grad():
                Mz = beta * Mz + M.grad
                bz = beta * bz + b.grad
                M -= alpha * Mz
                b -= alpha * bz
            M.grad.zero_()
            b.grad.zero_()
            if verbose >= 2:
                print(
                    f"epoch:{epoch}/{epochs} batch:{batch}/{cur_batches} obj:{obj:.5f}"
                )
            objs[batch] = obj.detach().numpy()

        last_obj = np.mean(objs)
        if verbose >= 1:
            print(f"epoch:{epoch} {objs[0]:.5f}->{objs[-1]:.5f} avg:{last_obj:.5f}")

        if epoch > 0 and last_obj < np.min(np.array(obj_history)):
            best_M = epoch_start_M
            best_b = epoch_start_b
        if epoch == full_epochs and full_epochs == 0:
            best_M = M.detach().numpy()
            best_b = b.detach().numpy()

        if len(obj_history) >= 10:
            if last_obj > np.max(np.array(obj_history[-10:])):
                # Terminate early if worse than all previous 10 iterations
                if verbose >= 1:
                    print(
                        f"Terminating {(alpha, beta)} after epoch {epoch}: {last_obj:.5f}"
                    )
                break
        obj_history.append(last_obj)
    if best_M.ndim == 1:
        best_M = np.diag(best_M)
    return (best_M, best_b, None)


class MMDAdapter:
    def __init__(
        self,
        transform_type: str = "location-scale",
        optim_kwargs: dict = None,
        verbose: Union[bool, int] = 1,
        debug: bool = False,
    ):
        """
        Args:
            transform_type: Whether to jointly transform all features ("affine"),
                or to transform each feature independently ("location-scale").
                Modeling the joint distribution is slower and allows each
                adapted feature to be a function of all other features,
                rather than purely based on that particular feature
                observation.

            optim_kwargs: Dict containing args for optimization.
                If mmd, valid keys are "epochs", "batch_size",
                "alpha" (learning rate), and "beta" (momentum).

            verbose: Bool or integer that indicates the verbosity.

            debug: Whether to save state for debugging.
        """
        if transform_type not in ("affine", "location-scale"):
            raise ValueError(f"invalid transform_type: {transform_type}")

        self.transform_type = transform_type
        self.optim_kwargs = deepcopy(optim_kwargs) or {}
        self.verbose = verbose
        self.debug = debug

    def fit(
        self,
        S: np.ndarray,
        T: np.ndarray,
    ):
        """

        Modifies M_, b_, possibly m_plot_, b_plot_, mb_objs_

        """
        num_S = S.shape[0]
        num_T = T.shape[0]
        assert S.shape[1] == T.shape[1]

        num_feats = S.shape[1]

        self.M_, self.b_, self.debug_dict_ = run_mmd(
            S=S,
            T=T,
            transform_type=self.transform_type,
            debug=self.debug,
            verbose=self.verbose,
            **self.optim_kwargs,
        )

        if self.transform_type == "location-scale":
            self.M_inv_ = np.diag(1.0 / np.diag(self.M_))
        elif self.transform_type == "affine":
            self.M_inv_ = np.linalg.inv(self.M_)

        return self

    def transform(
        self,
        S,
    ):

        # same as adaptedS = (self.M_ @ S.T).T + self.b_.reshape(1, -1)
        # self.b_.reshape(1, -1) has shape (1, num_feats)
        adaptedS = S @ self.M_.T + self.b_.reshape(1, -1)
        return adaptedS

    def inverse_transform(
        self,
        T,
    ):
        adaptedT = (T - self.b_.reshape(1, -1)) @ self.M_inv_.T
        return adaptedT
