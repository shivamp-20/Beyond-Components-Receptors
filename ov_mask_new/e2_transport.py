#!/usr/bin/env python3
"""E2: Linear transport in receptor space (Autonomous EDMD + DMDc).

Consumes Fix3 artifacts:
  outputs/<task>/artifacts/e1_fix3/

Writes:
  outputs/<task>/artifacts/e2_transport/

This version includes numerical fixes (ridge / sparse scaling, safe standardization)
and additional diagnostics + control sanity tests.

Convention (row snapshots):
  X: [M, C] where each row is x_t
  Y: [M, C] where each row is x_{t+1}
  Autonomous dynamics:   Y ≈ X W
  Stored operator A is transposed: A := W^T so prediction uses Yhat = X @ A.T

For DMDc:
  Omega = [X, U] where U: [M, d_u]
  Y ≈ Omega K  with K = [W_A; W_B]
  Stored A := W_A^T, B := W_B^T so Yhat = X @ A.T + U @ B.T
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# -------------------------
# Utilities
# -------------------------

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _spectral_radius_from_W(W: torch.Tensor) -> float:
    """Spectral radius of the exact operator used in rollout: x_{t+1} = x_t @ W."""
    Wn = _to_numpy(W)
    eig = np.linalg.eigvals(Wn)
    return float(np.max(np.abs(eig)))


def _stabilize_W(W: torch.Tensor, rho_target: float = 0.99) -> Tuple[torch.Tensor, Dict[str, float]]:
    rho_before = _spectral_radius_from_W(W)
    if rho_before > rho_target:
        scale = rho_target / (rho_before + 1e-12)
        W2 = W * scale
        rho_after = _spectral_radius_from_W(W2)
    else:
        scale = 1.0
        W2 = W
        rho_after = rho_before
    return W2, {
        "rho_before": rho_before,
        "rho_after": rho_after,
        "scale": float(scale),
    }


def nrmse_rows(Y: torch.Tensor, Yhat: torch.Tensor, eps: float = 1e-8) -> float:
    """Mean row-wise relative error: mean( ||e_i|| / (||y_i||+eps) )."""
    e = Yhat - Y
    num = torch.linalg.norm(e, dim=1)
    den = torch.linalg.norm(Y, dim=1) + eps
    return float(torch.mean(num / den).item())


def count_density(W: torch.Tensor, thr: float = 1e-6) -> Tuple[int, float]:
    nnz = int((W.abs() > thr).sum().item())
    total = W.numel()
    return nnz, nnz / float(total)


def topk_edges(W: torch.Tensor, k: int = 200) -> List[Tuple[int, int, float]]:
    """Return list of (dst_i, src_j, weight) for top-k |W_ij| edges."""
    absW = W.abs().flatten()
    k = min(k, absW.numel())
    vals, idx = torch.topk(absW, k)
    out: List[Tuple[int, int, float]] = []
    C_out, C_in = W.shape
    for v, flat in zip(vals.tolist(), idx.tolist()):
        i = flat // C_in
        j = flat % C_in
        out.append((int(i), int(j), float(W[i, j].item())))
    return out


# -------------------------
# Data structures
# -------------------------

@dataclass
class SplitData:
    X: torch.Tensor          # [M,C]
    Y: torch.Tensor          # [M,C]
    U: Optional[torch.Tensor]  # [M,du] or None
    X_seq: torch.Tensor      # [N,L,C]
    U_seq: Optional[torch.Tensor]  # [N,L,du] or None


@dataclass
class Standardizer:
    mu_x: torch.Tensor
    sig_x: torch.Tensor
    mu_u: Optional[torch.Tensor] = None
    sig_u: Optional[torch.Tensor] = None

    def x(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.mu_x) / self.sig_x

    def y(self, Y: torch.Tensor) -> torch.Tensor:
        # same state space
        return (Y - self.mu_x) / self.sig_x

    def u(self, U: torch.Tensor) -> torch.Tensor:
        if self.mu_u is None or self.sig_u is None:
            # fallback (shouldn't happen if DMDc used)
            return U / self.sig_x
        return (U - self.mu_u) / self.sig_u


def fit_standardizer(X_train: torch.Tensor,
                     U_train: Optional[torch.Tensor],
                     eps: float = 1e-6) -> Standardizer:
    """Fit standardization statistics on TRAIN only."""
    mu_x = X_train.mean(dim=0)
    sig_x = X_train.std(dim=0)
    sig_x = torch.clamp(sig_x, min=eps)

    if U_train is None:
        return Standardizer(mu_x=mu_x, sig_x=sig_x)

    mu_u = U_train.mean(dim=0)
    sig_u = U_train.std(dim=0)
    sig_u = torch.clamp(sig_u, min=eps)
    return Standardizer(mu_x=mu_x, sig_x=sig_x, mu_u=mu_u, sig_u=sig_u)


# -------------------------
# IO
# -------------------------


def load_fix3(base_dir: str, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load X_clean and X_corr from Fix3 artifacts."""
    x_clean_p = os.path.join(base_dir, f"X_clean_{split}_fix3.pt")
    x_corr_p  = os.path.join(base_dir, f"X_corr_{split}_fix3.pt")
    if not os.path.exists(x_clean_p):
        raise FileNotFoundError(f"Missing {x_clean_p}")
    if not os.path.exists(x_corr_p):
        raise FileNotFoundError(f"Missing {x_corr_p}")
    Xc = torch.load(x_clean_p, map_location="cpu")
    Xr = torch.load(x_corr_p, map_location="cpu")
    if not isinstance(Xc, torch.Tensor) or not isinstance(Xr, torch.Tensor):
        raise TypeError("Fix3 X tensors must be torch.Tensor")
    return Xc, Xr


def build_split_data(Xc: torch.Tensor, Xr: torch.Tensor) -> SplitData:
    """Build snapshot pairs (X,Y) and control U = Xcorr - Xclean."""
    if Xc.ndim != 3:
        raise ValueError(f"Expected Xc [N,L,C], got {tuple(Xc.shape)}")
    if Xr.shape != Xc.shape:
        raise ValueError(f"Xr shape {tuple(Xr.shape)} != Xc shape {tuple(Xc.shape)}")

    N, L, C = Xc.shape
    M = N * (L - 1)

    X0 = Xc[:, 0:L-1, :].reshape(M, C)
    Y0 = Xc[:, 1:L,   :].reshape(M, C)
    U0 = (Xr - Xc)[:, 0:L-1, :].reshape(M, C)

    return SplitData(X=X0, Y=Y0, U=U0, X_seq=Xc, U_seq=(Xr - Xc))


# -------------------------
# Solvers (Ridge)
# -------------------------


def ridge_autonomous_W(X: torch.Tensor, Y: torch.Tensor, lam: float) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Solve for W in Y ≈ X W with ridge: (1/2M)||XW-Y||^2 + (lam/2)||W||^2.

    Returns W (mapping matrix) in float32, and diagnostics.
    """
    X64 = X.double()
    Y64 = Y.double()
    M = X64.shape[0]
    C = X64.shape[1]

    XtX = (X64.T @ X64) / M
    XtY = (X64.T @ Y64) / M
    I = torch.eye(C, dtype=torch.float64)
    G = XtX + lam * I

    # Condition number via eigenvalues (G is SPD if lam>0)
    try:
        evals = torch.linalg.eigvalsh(G)
        cond = float((evals.max() / torch.clamp(evals.min(), min=1e-18)).item())
    except Exception:
        cond = float("nan")

    W64 = torch.linalg.solve(G, XtY)
    W = W64.float()

    diag = {
        "lam": float(lam),
        "cond_G": cond,
        "max_abs_W": float(W64.abs().max().item()),
        "fro_W": float(torch.linalg.norm(W64).item()),
        "base_diag_xtx": float(torch.mean(torch.diagonal(XtX)).item()),
    }
    return W, diag


def ridge_dmdc_W(X: torch.Tensor, U: torch.Tensor, Y: torch.Tensor, lam: float) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Solve DMDc ridge: Y ≈ [X,U] K with ridge, returns (W_A, W_B)."""
    X64 = X.double()
    U64 = U.double()
    Y64 = Y.double()

    M = X64.shape[0]
    C = X64.shape[1]
    du = U64.shape[1]

    Omega = torch.cat([X64, U64], dim=1)  # [M, C+du]
    OtO = (Omega.T @ Omega) / M
    OtY = (Omega.T @ Y64) / M

    D = C + du
    I = torch.eye(D, dtype=torch.float64)
    G = OtO + lam * I

    try:
        evals = torch.linalg.eigvalsh(G)
        cond = float((evals.max() / torch.clamp(evals.min(), min=1e-18)).item())
    except Exception:
        cond = float("nan")

    K64 = torch.linalg.solve(G, OtY)  # [D, C]
    WA = K64[:C, :].float()          # [C,C] mapping
    WB = K64[C:, :].float()          # [du,C] mapping

    diag = {
        "lam": float(lam),
        "cond_G": cond,
        "max_abs_WA": float(K64[:C, :].abs().max().item()),
        "max_abs_WB": float(K64[C:, :].abs().max().item()),
        "fro_WA": float(torch.linalg.norm(K64[:C, :]).item()),
        "fro_WB": float(torch.linalg.norm(K64[C:, :]).item()),
    }
    return WA, WB, diag


# -------------------------
# Solvers (Sparse FISTA)
# -------------------------


def _soft_threshold(X: torch.Tensor, thr: float) -> torch.Tensor:
    return torch.sign(X) * torch.clamp(torch.abs(X) - thr, min=0.0)


def fista_autonomous_W_from_gram(
    XtX: torch.Tensor,  # [C,C] float64
    XtY: torch.Tensor,  # [C,C] float64
    W0: torch.Tensor,   # [C,C] float64
    l1: float,
    max_iters: int = 2000,
    tol: float = 1e-6,
    eval_every: int = 50,
    patience: int = 6,
    X_val: Optional[torch.Tensor] = None,
    Y_val: Optional[torch.Tensor] = None,
    density_thr: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """FISTA for min_W 0.5*tr(W^T XtX W) - tr(W^T XtY) + l1*||W||_1.

    XtX and XtY are already scaled by 1/M.
    """
    # Lipschitz constant is max eigenvalue of XtX
    eigs = torch.linalg.eigvalsh(XtX)
    L = float(torch.clamp(eigs.max(), min=1e-18).item())
    eta = 1.0 / (L + 1e-12)

    W = W0.clone()
    Z = W0.clone()
    t = 1.0

    def obj(W_: torch.Tensor) -> float:
        # smooth part up to constant
        smooth = 0.5 * torch.sum(W_ * (XtX @ W_)) - torch.sum(W_ * XtY)
        l1term = l1 * torch.sum(torch.abs(W_))
        return float((smooth + l1term).item())

    best_val = float("inf")
    best_W = W.clone()
    no_improve = 0
    prev_obj = obj(W)

    for it in range(1, max_iters + 1):
        grad = XtX @ Z - XtY
        W_next = _soft_threshold(Z - eta * grad, eta * l1)

        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        Z = W_next + ((t - 1.0) / t_next) * (W_next - W)
        W = W_next
        t = t_next

        if it % eval_every == 0 or it == max_iters:
            cur_obj = obj(W)
            rel = abs(prev_obj - cur_obj) / (abs(prev_obj) + 1e-12)
            prev_obj = cur_obj

            # optional val check
            if X_val is not None and Y_val is not None:
                Yhat = X_val @ W.float()
                val_n = nrmse_rows(Y_val, Yhat)
                if val_n + 1e-9 < best_val:
                    best_val = val_n
                    best_W = W.clone()
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= patience and rel < tol:
                    break
            else:
                if rel < tol:
                    break

    W_final = best_W if X_val is not None else W
    nnz, dens = count_density(W_final.float(), thr=density_thr)
    stats = {
        "l1": float(l1),
        "iters": int(it),
        "eta": float(eta),
        "L": float(L),
        "val_best": float(best_val) if X_val is not None else float("nan"),
        "nnz": int(nnz),
        "density": float(dens),
        "max_abs_W": float(W_final.abs().max().item()),
        "fro_W": float(torch.linalg.norm(W_final).item()),
    }
    return W_final, stats


def fista_dmdc_K_from_gram(
    OtO: torch.Tensor,  # [D,D] float64
    OtY: torch.Tensor,  # [D,C] float64
    K0: torch.Tensor,   # [D,C] float64
    l1_A: float,
    C_state: int,
    max_iters: int = 2000,
    tol: float = 1e-6,
    eval_every: int = 50,
    patience: int = 6,
    X_val: Optional[torch.Tensor] = None,
    U_val: Optional[torch.Tensor] = None,
    Y_val: Optional[torch.Tensor] = None,
    density_thr: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Sparse DMDc via FISTA on K with L1 applied only to state block rows (0:C_state).

    Objective: 0.5*tr(K^T OtO K) - tr(K^T OtY) + l1_A*||K_state||_1
    where K_state = K[0:C_state, :].
    """
    eigs = torch.linalg.eigvalsh(OtO)
    L = float(torch.clamp(eigs.max(), min=1e-18).item())
    eta = 1.0 / (L + 1e-12)

    K = K0.clone()
    Z = K0.clone()
    t = 1.0

    def prox(K_: torch.Tensor) -> torch.Tensor:
        K2 = K_.clone()
        K2[:C_state, :] = _soft_threshold(K2[:C_state, :], eta * l1_A)
        return K2

    def obj(K_: torch.Tensor) -> float:
        smooth = 0.5 * torch.sum(K_ * (OtO @ K_)) - torch.sum(K_ * OtY)
        l1term = l1_A * torch.sum(torch.abs(K_[:C_state, :]))
        return float((smooth + l1term).item())

    best_val = float("inf")
    best_K = K.clone()
    no_improve = 0
    prev_obj = obj(K)

    for it in range(1, max_iters + 1):
        grad = OtO @ Z - OtY
        K_next = prox(Z - eta * grad)

        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        Z = K_next + ((t - 1.0) / t_next) * (K_next - K)
        K = K_next
        t = t_next

        if it % eval_every == 0 or it == max_iters:
            cur_obj = obj(K)
            rel = abs(prev_obj - cur_obj) / (abs(prev_obj) + 1e-12)
            prev_obj = cur_obj

            if X_val is not None and U_val is not None and Y_val is not None:
                C = X_val.shape[1]
                WA = K[:C, :].float()
                WB = K[C:, :].float()
                Yhat = X_val @ WA + U_val @ WB
                val_n = nrmse_rows(Y_val, Yhat)
                if val_n + 1e-9 < best_val:
                    best_val = val_n
                    best_K = K.clone()
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= patience and rel < tol:
                    break
            else:
                if rel < tol:
                    break

    K_final = best_K if X_val is not None else K

    # density counted on WA only (receptor->receptor edges)
    C = C_state
    WA = K_final[:C, :].float().T  # A stored as transpose later; but for density we look at WA mapping
    nnz, dens = count_density(WA, thr=density_thr)

    stats = {
        "l1_A": float(l1_A),
        "iters": int(it),
        "eta": float(eta),
        "L": float(L),
        "val_best": float(best_val) if X_val is not None else float("nan"),
        "nnz_A": int(nnz),
        "density_A": float(dens),
        "max_abs_K": float(K_final.abs().max().item()),
    }
    return K_final, stats


# -------------------------
# Rollout evaluation
# -------------------------


def rollout_nrmse_by_layer_autonomous(
    X_seq_std: torch.Tensor,  # [N,L,C] standardized
    A: torch.Tensor,          # stored A (so W=A.T)
    eps: float = 1e-8,
) -> Tuple[float, List[float]]:
    """Roll out x_{t+1} = x_t @ A.T in standardized space."""
    N, L, C = X_seq_std.shape
    W = A.T

    xhat = X_seq_std[:, 0, :].clone()  # [N,C]
    errs: List[float] = []
    # include error at each layer t=1..L-1
    for t in range(L - 1):
        xhat = xhat @ W
        true = X_seq_std[:, t + 1, :]
        num = torch.linalg.norm(xhat - true, dim=1)
        den = torch.linalg.norm(true, dim=1) + eps
        errs.append(float(torch.mean(num / den).item()))
    return float(np.mean(errs)), errs


# -------------------------
# Plot helpers
# -------------------------


def _save_json(path: str, obj) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _save_csv(path: str, header: List[str], rows: List[List]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _try_plot_eigs(eigs: np.ndarray, out_png: str) -> None:
    try:
        import matplotlib.pyplot as plt
        ensure_dir(os.path.dirname(out_png))
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(eigs.real, eigs.imag, s=10)
        # unit circle
        th = np.linspace(0, 2*np.pi, 400)
        ax.plot(np.cos(th), np.sin(th), linewidth=1)
        ax.set_xlabel("Re")
        ax.set_ylabel("Im")
        ax.set_title("Eigenvalues")
        ax.set_aspect('equal', 'box')
        fig.tight_layout()
        fig.savefig(out_png, dpi=160)
        plt.close(fig)
    except Exception:
        pass


def _try_plot_rollout_curves(err_by_layer: Dict[str, List[float]], out_png: str) -> None:
    try:
        import matplotlib.pyplot as plt
        ensure_dir(os.path.dirname(out_png))
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(1, 1, 1)
        for name, arr in err_by_layer.items():
            ax.plot(np.arange(1, len(arr)+1), arr, label=name)
        ax.set_xlabel("Layer t")
        ax.set_ylabel("NRMSE")
        ax.legend()
        ax.set_title("Rollout error by layer")
        fig.tight_layout()
        fig.savefig(out_png, dpi=160)
        plt.close(fig)
    except Exception:
        pass


def _try_plot_edge_mass_cdf(W: torch.Tensor, out_png: str) -> None:
    try:
        import matplotlib.pyplot as plt
        ensure_dir(os.path.dirname(out_png))
        w = W.abs().flatten().detach().cpu().numpy()
        w = np.sort(w)[::-1]
        if w.size == 0:
            return
        cdf = np.cumsum(w) / (np.sum(w) + 1e-12)
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(np.arange(1, len(cdf)+1), cdf)
        ax.set_xlabel("#edges (sorted by |w|)")
        ax.set_ylabel("Cumulative |w| mass")
        ax.set_title("Edge mass CDF")
        fig.tight_layout()
        fig.savefig(out_png, dpi=160)
        plt.close(fig)
    except Exception:
        pass


# -------------------------
# Main
# -------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, required=True, choices=["gp", "ioi", "gt"])
    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])
    p.add_argument("--standardize", action="store_true", default=True)
    p.add_argument("--no_standardize", dest="standardize", action="store_false")
    p.add_argument("--eps", type=float, default=1e-6)

    # Ridge grid: 'auto' => base * 10**linspace(-6,2,25)
    p.add_argument("--ridge_lambdas", type=str, default="auto")

    # Sparse grid: 'auto' => lam1_max * 10**linspace(-3,-0.5,12)
    p.add_argument("--l1_lambdas", type=str, default="auto")
    p.add_argument("--fista_iters", type=int, default=2000)
    p.add_argument("--fista_tol", type=float, default=1e-6)

    p.add_argument("--apply_stability_scaling", action="store_true", default=True)
    p.add_argument("--rho_target", type=float, default=0.99)

    p.add_argument("--edge_thr", type=float, default=1e-4)  # for reporting
    p.add_argument("--density_thr", type=float, default=1e-6)  # for selection counting
    p.add_argument("--top_edge_k", type=int, default=200)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shuffles", type=int, default=20)

    return p.parse_args()


def _parse_list_or_auto(s: str) -> Optional[List[float]]:
    s = (s or "").strip().lower()
    if s == "auto" or s == "":
        return None
    parts = [x.strip() for x in s.replace(";", ",").split(",") if x.strip()]
    return [float(x) for x in parts]


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # device selection (mostly CPU is enough because we use Gram matrices)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    out_dir = args.out_dir
    base_fix3 = os.path.join(out_dir, "artifacts", "e1_fix3")
    out_e2 = os.path.join(out_dir, "artifacts", "e2_transport")

    sub_models = ensure_dir(os.path.join(out_e2, "models"))
    sub_metrics = ensure_dir(os.path.join(out_e2, "metrics"))
    sub_plots = ensure_dir(os.path.join(out_e2, "plots"))
    sub_tables = ensure_dir(os.path.join(out_e2, "tables"))
    sub_debug = ensure_dir(os.path.join(out_e2, "debug"))

    # Load bank (for metadata in tables)
    bank_path = os.path.join(base_fix3, "receptor_bank.pt")
    if not os.path.exists(bank_path):
        raise FileNotFoundError(f"Missing {bank_path}")
    bank = torch.load(bank_path, map_location="cpu")
    rep_meta = bank.get("rep_meta", None)

    # Load splits
    Xc_tr, Xr_tr = load_fix3(base_fix3, "train")
    Xc_va, Xr_va = load_fix3(base_fix3, "val")
    Xc_te, Xr_te = load_fix3(base_fix3, "test")

    train = build_split_data(Xc_tr, Xr_tr)
    val = build_split_data(Xc_va, Xr_va)
    test = build_split_data(Xc_te, Xr_te)

    N_train, L, C = train.X_seq.shape
    N_val = val.X_seq.shape[0]
    N_test = test.X_seq.shape[0]

    # sanity prints
    M = train.X.shape[0]
    print(f"[E2] task={args.task} C={C} L={L} N_train={N_train} N_val={N_val} N_test={N_test}")
    print(f"[E2] shapes: X0_train={tuple(train.X.shape)} Y0_train={tuple(train.Y.shape)} U0_train={tuple(train.U.shape) if train.U is not None else None}  M={M}")

    # Standardize
    if args.standardize:
        std = fit_standardizer(train.X, train.U, eps=args.eps)
        sig_min = float(std.sig_x.min().item())
        sig_med = float(std.sig_x.median().item())
        sig_max = float(std.sig_x.max().item())
        print(f"[E2] sig_x stats: min={sig_min:.3e} median={sig_med:.3e} max={sig_max:.3e}")
        Xtr = std.x(train.X)
        Ytr = std.y(train.Y)
        Utr = std.u(train.U)
        Xva = std.x(val.X)
        Yva = std.y(val.Y)
        Uva = std.u(val.U)
        Xte = std.x(test.X)
        Yte = std.y(test.Y)
        Ute = std.u(test.U)

        Xseq_tr = std.x(train.X_seq)
        Xseq_va = std.x(val.X_seq)
        Xseq_te = std.x(test.X_seq)

        print(f"[E2] X std diagnostics: max|Xtr|={float(Xtr.abs().max().item()):.3e} mean|Xtr|={float(Xtr.abs().mean().item()):.3e}")
        if std.sig_u is not None:
            print(f"[E2] sig_u stats: min={float(std.sig_u.min().item()):.3e} median={float(std.sig_u.median().item()):.3e} max={float(std.sig_u.max().item()):.3e}")
    else:
        std = None
        Xtr, Ytr, Utr = train.X, train.Y, train.U
        Xva, Yva, Uva = val.X, val.Y, val.U
        Xte, Yte, Ute = test.X, test.Y, test.U
        Xseq_tr, Xseq_va, Xseq_te = train.X_seq, val.X_seq, test.X_seq

    # Move to chosen device for speed (still OK to keep on CPU)
    Xtr = Xtr.to(device)
    Ytr = Ytr.to(device)
    Xva = Xva.to(device)
    Yva = Yva.to(device)
    Xte = Xte.to(device)
    Yte = Yte.to(device)
    Utr = Utr.to(device)
    Uva = Uva.to(device)
    Ute = Ute.to(device)

    # -------------------------
    # Baselines
    # -------------------------

    # identity
    W_id = torch.eye(C, dtype=torch.float32, device=device)
    A_id = W_id.T

    # diagonal
    # fit per-dimension in standardized space
    eps = args.eps
    denom = (Xtr * Xtr).sum(dim=0) + eps
    a = (Xtr * Ytr).sum(dim=0) / denom
    W_diag = torch.diag(a)
    A_diag = W_diag.T

    # evaluate baselines
    val_1 = {
        "identity": nrmse_rows(Yva, Xva @ W_id),
        "diag": nrmse_rows(Yva, Xva @ W_diag),
    }
    test_1 = {
        "identity": nrmse_rows(Yte, Xte @ W_id),
        "diag": nrmse_rows(Yte, Xte @ W_diag),
    }

    # -------------------------
    # Ridge (Autonomous)
    # -------------------------

    # build XtX, XtY once in float64 on CPU (stable)
    Xtr_cpu64 = Xtr.detach().cpu().double()
    Ytr_cpu64 = Ytr.detach().cpu().double()
    Mtr = Xtr_cpu64.shape[0]
    XtX = (Xtr_cpu64.T @ Xtr_cpu64) / Mtr
    XtY = (Xtr_cpu64.T @ Ytr_cpu64) / Mtr
    base = float(torch.mean(torch.diagonal(XtX)).item())

    ridge_user = _parse_list_or_auto(args.ridge_lambdas)
    if ridge_user is None:
        exps = np.linspace(-6, 2, 25)
        ridge_lams = [base * float(10.0 ** e) for e in exps]
    else:
        # treat user lambdas as absolute in standardized space
        ridge_lams = ridge_user

    sweep_rows: List[List] = []
    best_val = float("inf")
    best_W_ridge = None
    best_ridge_diag = None

    for lam in ridge_lams:
        W, diag = ridge_autonomous_W(Xtr.detach().cpu(), Ytr.detach().cpu(), lam=float(lam))
        # val metric computed on device
        Wd = W.to(device)
        v = nrmse_rows(Yva, Xva @ Wd)
        nnz, dens = count_density(W, thr=args.density_thr)
        sweep_rows.append([lam, v, dens, diag.get("cond_G", float("nan")), diag.get("max_abs_W", float("nan"))])
        if v < best_val:
            best_val = v
            best_W_ridge = W
            best_ridge_diag = diag

    if best_W_ridge is None:
        raise RuntimeError("Ridge sweep produced no candidate")

    W_ridge = best_W_ridge.to(device)

    # optional stability scaling on W (rollout uses W)
    stab_ridge = {}
    if args.apply_stability_scaling:
        W_ridge2, stab_ridge = _stabilize_W(W_ridge.detach().cpu(), rho_target=args.rho_target)
        W_ridge = W_ridge2.to(device)

    A_ridge = W_ridge.T

    val_1["ridge"] = nrmse_rows(Yva, Xva @ W_ridge)
    test_1["ridge"] = nrmse_rows(Yte, Xte @ W_ridge)

    torch.save({"A_std": A_ridge.detach().cpu(), "W_std": W_ridge.detach().cpu(), "ridge_diag": best_ridge_diag, "stability": stab_ridge},
               os.path.join(sub_models, "A_autonomous_ridge.pt"))

    _save_csv(os.path.join(sub_tables, "lambda_sweep_autonomous.csv"),
              ["lambda", "val_nrmse", "density", "cond_G", "max_abs_W"],
              sweep_rows)

    # -------------------------
    # Sparse (Autonomous)
    # -------------------------

    l1_user = _parse_list_or_auto(args.l1_lambdas)
    if l1_user is None:
        lam1_max = float(torch.max(torch.abs(XtY)).item())
        exps = np.linspace(-3, -0.5, 12)
        l1_lams = [lam1_max * float(10.0 ** e) for e in exps]
        l1_lams = [0.0] + l1_lams
    else:
        l1_lams = l1_user

    sparse_sweep: List[List] = []
    best_sparse = None
    best_sparse_val = float("inf")
    best_sparse_stats = None

    # warm start from ridge in float64
    W0 = best_W_ridge.double()

    for l1 in l1_lams:
        W64, st = fista_autonomous_W_from_gram(
            XtX=XtX,
            XtY=XtY,
            W0=W0,
            l1=float(l1),
            max_iters=args.fista_iters,
            tol=args.fista_tol,
            eval_every=50,
            patience=6,
            X_val=Xva.detach().cpu(),
            Y_val=Yva.detach().cpu(),
            density_thr=args.density_thr,
        )
        W = W64.float().to(device)
        v = nrmse_rows(Yva, Xva @ W)
        nnz, dens = count_density(W64.float(), thr=args.density_thr)
        sparse_sweep.append([l1, v, dens, st.get("iters", None), st.get("max_abs_W", None)])

        # selection rule: lowest val NRMSE subject to density<=0.10
        if dens <= 0.10 + 1e-12:
            if v < best_sparse_val:
                best_sparse_val = v
                best_sparse = W64
                best_sparse_stats = st
        else:
            # if nothing meets density, still track best overall
            if best_sparse is None and v < best_sparse_val:
                best_sparse_val = v
                best_sparse = W64
                best_sparse_stats = st

    if best_sparse is None:
        best_sparse = W0
        best_sparse_stats = {"note": "fallback_to_ridge_warm_start"}

    W_sparse = best_sparse.float().to(device)

    stab_sparse = {}
    if args.apply_stability_scaling:
        W_sparse2, stab_sparse = _stabilize_W(W_sparse.detach().cpu(), rho_target=args.rho_target)
        W_sparse = W_sparse2.to(device)

    A_sparse = W_sparse.T

    val_1["sparse"] = nrmse_rows(Yva, Xva @ W_sparse)
    test_1["sparse"] = nrmse_rows(Yte, Xte @ W_sparse)

    torch.save({
        "A_std": A_sparse.detach().cpu(),
        "W_std": W_sparse.detach().cpu(),
        "best_l1": float(best_sparse_stats.get("l1", 0.0)) if isinstance(best_sparse_stats, dict) else None,
        "sweep": sparse_sweep,
        "stability": stab_sparse,
    }, os.path.join(sub_models, "A_autonomous_sparse.pt"))

    _save_csv(os.path.join(sub_tables, "lambda_sweep_autonomous_l1.csv"),
              ["l1", "val_nrmse", "density", "iters", "max_abs_W"],
              sparse_sweep)

    # -------------------------
    # DMDc ridge / sparse
    # -------------------------

    # build Omega grams on CPU float64
    Utr_cpu64 = Utr.detach().cpu().double()
    Omega_tr = torch.cat([Xtr_cpu64, Utr_cpu64], dim=1)  # [M, 2C]
    OtO = (Omega_tr.T @ Omega_tr) / Mtr
    OtY = (Omega_tr.T @ Ytr_cpu64) / Mtr

    # DMDc ridge sweep uses same ridge_lams
    dmdc_sweep: List[List] = []
    best_val_d = float("inf")
    best_WA = None
    best_WB = None
    best_d_diag = None

    for lam in ridge_lams:
        WA, WB, diag = ridge_dmdc_W(Xtr.detach().cpu(), Utr.detach().cpu(), Ytr.detach().cpu(), lam=float(lam))
        WAd = WA.to(device)
        WBd = WB.to(device)
        v = nrmse_rows(Yva, (Xva @ WAd) + (Uva @ WBd))
        dmdc_sweep.append([lam, v, diag.get("cond_G", float("nan")), diag.get("max_abs_WA", float("nan")), diag.get("max_abs_WB", float("nan"))])
        if v < best_val_d:
            best_val_d = v
            best_WA = WA
            best_WB = WB
            best_d_diag = diag

    if best_WA is None or best_WB is None:
        raise RuntimeError("DMDc ridge sweep produced no candidate")

    WA_ridge = best_WA.to(device)
    WB_ridge = best_WB.to(device)

    # Stabilize ONLY state dynamics (WA)
    stab_d_ridge = {}
    if args.apply_stability_scaling:
        WA2, stab_d_ridge = _stabilize_W(WA_ridge.detach().cpu(), rho_target=args.rho_target)
        WA_ridge = WA2.to(device)

    A_dmdc_ridge = WA_ridge.T
    B_dmdc_ridge = WB_ridge.T

    val_1["dmdc_ridge"] = nrmse_rows(Yva, (Xva @ WA_ridge) + (Uva @ WB_ridge))
    test_1["dmdc_ridge"] = nrmse_rows(Yte, (Xte @ WA_ridge) + (Ute @ WB_ridge))

    torch.save({
        "A_std": A_dmdc_ridge.detach().cpu(),
        "B_std": B_dmdc_ridge.detach().cpu(),
        "WA_std": WA_ridge.detach().cpu(),
        "WB_std": WB_ridge.detach().cpu(),
        "ridge_diag": best_d_diag,
        "stability": stab_d_ridge,
    }, os.path.join(sub_models, "A_B_dmdc_ridge.pt"))

    _save_csv(os.path.join(sub_tables, "lambda_sweep_dmdc.csv"),
              ["lambda", "val_nrmse", "cond_G", "max_abs_WA", "max_abs_WB"],
              dmdc_sweep)

    # DMDc sparse: penalize only WA (state block)
    dmdc_l1_sweep: List[List] = []
    best_val_ds = float("inf")
    best_K = None
    best_K_stats = None

    # warm-start K0 from ridge (float64)
    K0 = torch.cat([best_WA.double(), best_WB.double()], dim=0)  # [2C, C]

    for l1 in l1_lams:
        K64, st = fista_dmdc_K_from_gram(
            OtO=OtO,
            OtY=OtY,
            K0=K0,
            l1_A=float(l1),
            C_state=C,
            max_iters=args.fista_iters,
            tol=args.fista_tol,
            eval_every=50,
            patience=6,
            X_val=Xva.detach().cpu(),
            U_val=Uva.detach().cpu(),
            Y_val=Yva.detach().cpu(),
            density_thr=args.density_thr,
        )
        WA = K64[:C, :].float().to(device)
        WB = K64[C:, :].float().to(device)
        v = nrmse_rows(Yva, (Xva @ WA) + (Uva @ WB))
        # density on WA only
        nnz, dens = count_density(K64[:C, :].float(), thr=args.density_thr)
        dmdc_l1_sweep.append([l1, v, dens, st.get("iters", None), st.get("max_abs_K", None)])

        if dens <= 0.10 + 1e-12:
            if v < best_val_ds:
                best_val_ds = v
                best_K = K64
                best_K_stats = st
        else:
            if best_K is None and v < best_val_ds:
                best_val_ds = v
                best_K = K64
                best_K_stats = st

    if best_K is None:
        best_K = K0
        best_K_stats = {"note": "fallback_to_ridge_warm_start"}

    WA_sparse = best_K[:C, :].float().to(device)
    WB_sparse = best_K[C:, :].float().to(device)

    stab_d_sparse = {}
    if args.apply_stability_scaling:
        WA2, stab_d_sparse = _stabilize_W(WA_sparse.detach().cpu(), rho_target=args.rho_target)
        WA_sparse = WA2.to(device)

    A_dmdc_sparse = WA_sparse.T
    B_dmdc_sparse = WB_sparse.T

    val_1["dmdc_sparse"] = nrmse_rows(Yva, (Xva @ WA_sparse) + (Uva @ WB_sparse))
    test_1["dmdc_sparse"] = nrmse_rows(Yte, (Xte @ WA_sparse) + (Ute @ WB_sparse))

    torch.save({
        "A_std": A_dmdc_sparse.detach().cpu(),
        "B_std": B_dmdc_sparse.detach().cpu(),
        "WA_std": WA_sparse.detach().cpu(),
        "WB_std": WB_sparse.detach().cpu(),
        "best_l1_A": float(best_K_stats.get("l1_A", 0.0)) if isinstance(best_K_stats, dict) else None,
        "sweep": dmdc_l1_sweep,
        "stability": stab_d_sparse,
    }, os.path.join(sub_models, "A_B_dmdc_sparse.pt"))

    _save_csv(os.path.join(sub_tables, "lambda_sweep_dmdc_l1.csv"),
              ["l1_A", "val_nrmse", "density_A", "iters", "max_abs_K"],
              dmdc_l1_sweep)

    # -------------------------
    # Rollout evaluation (Autonomous models)
    # -------------------------

    # Rollouts must use standardized sequences and the exact operator used in rollout.
    # Build A matrices for identity/diag/ridge/sparse
    models_auto = {
        "identity": A_id.detach().cpu(),
        "diag": A_diag.detach().cpu(),
        "ridge": A_ridge.detach().cpu(),
        "sparse": A_sparse.detach().cpu(),
    }

    rollout_val = {}
    rollout_test = {}
    rollout_by_layer_val: Dict[str, List[float]] = {}

    for name, A in models_auto.items():
        tot, by = rollout_nrmse_by_layer_autonomous(Xseq_va.detach().cpu(), A, eps=args.eps)
        rollout_val[name] = tot
        rollout_by_layer_val[name] = by
        tot2, _ = rollout_nrmse_by_layer_autonomous(Xseq_te.detach().cpu(), A, eps=args.eps)
        rollout_test[name] = tot2

    np.save(os.path.join(sub_debug, "rollout_err_by_layer_val.npy"), rollout_by_layer_val)
    _try_plot_rollout_curves(rollout_by_layer_val, os.path.join(sub_plots, "rollout_error_curve_val.png"))

    # -------------------------
    # Control sanity
    # -------------------------

    # Compare with-U vs no-U using *selected* DMDc sparse (or ridge if sparse not better)
    # We'll use the best of (dmdc_sparse, dmdc_ridge) on val.
    use_sparse = val_1["dmdc_sparse"] <= val_1["dmdc_ridge"]
    if use_sparse:
        WA_use = WA_sparse.detach().cpu()
        WB_use = WB_sparse.detach().cpu()
        tag = "dmdc_sparse"
    else:
        WA_use = WA_ridge.detach().cpu()
        WB_use = WB_ridge.detach().cpu()
        tag = "dmdc_ridge"

    Yhat_withU = (Xva.detach().cpu() @ WA_use) + (Uva.detach().cpu() @ WB_use)
    Yhat_noU = (Xva.detach().cpu() @ WA_use)
    n_with = nrmse_rows(Yva.detach().cpu(), Yhat_withU)
    n_noU = nrmse_rows(Yva.detach().cpu(), Yhat_noU)

    # Shuffled-control refits (ridge only, fast + stable)
    shuf_vals: List[float] = []
    rng = np.random.RandomState(args.seed)
    best_lam_d = float(dmdc_sweep[int(np.argmin([r[1] for r in dmdc_sweep]))][0])

    Xtr_cpu = Xtr.detach().cpu()
    Ytr_cpu = Ytr.detach().cpu()
    Utr_cpu = Utr.detach().cpu()

    for s in range(args.shuffles):
        perm = rng.permutation(Utr_cpu.shape[0])
        Utr_shuf = Utr_cpu[perm]
        WA_sh, WB_sh, _ = ridge_dmdc_W(Xtr_cpu, Utr_shuf, Ytr_cpu, lam=best_lam_d)
        Yhat = (Xva.detach().cpu() @ WA_sh) + (Uva.detach().cpu() @ WB_sh)
        shuf_vals.append(nrmse_rows(Yva.detach().cpu(), Yhat))

    shuf_mean = float(np.mean(shuf_vals))
    shuf_std = float(np.std(shuf_vals))

    control_sanity = {
        "picked": tag,
        "val_nrmse_withU": float(n_with),
        "val_nrmse_noU": float(n_noU),
        "delta_noU_minus_withU": float(n_noU - n_with),
        "best_lambda_dmdc_ridge": float(best_lam_d),
        "shuffled_trainU_val_eval_mean": shuf_mean,
        "shuffled_trainU_val_eval_std": shuf_std,
        "mean_control_norm_train": float(torch.linalg.norm(train.U, dim=1).mean().item()),
        "mean_control_norm_val": float(torch.linalg.norm(val.U, dim=1).mean().item()),
    }

    # -------------------------
    # Interpretability outputs (sparse autonomous)
    # -------------------------

    # Graph stats for autonomous sparse model (on W)
    W_sparse_cpu = W_sparse.detach().cpu()
    nnz_report = int((W_sparse_cpu.abs() > args.edge_thr).sum().item())
    density_report = nnz_report / float(W_sparse_cpu.numel())

    abs_sorted = torch.sort(W_sparse_cpu.abs().flatten(), descending=True).values
    total_mass = float(abs_sorted.sum().item()) + 1e-12
    cdf = torch.cumsum(abs_sorted, dim=0) / total_mass

    def mass_frac(k: int) -> float:
        if k <= 0:
            return 0.0
        k = min(k, cdf.numel())
        return float(cdf[k-1].item())

    graph_stats = {
        "edge_thr": float(args.edge_thr),
        "nnz": int(nnz_report),
        "density": float(density_report),
        "top10_edge_mass_frac": mass_frac(10),
        "top50_edge_mass_frac": mass_frac(50),
        "top200_edge_mass_frac": mass_frac(200),
        "spectral_radius_W_sparse": float(_spectral_radius_from_W(W_sparse_cpu)),
        "max_abs_W_sparse": float(W_sparse_cpu.abs().max().item()),
    }

    _try_plot_edge_mass_cdf(W_sparse_cpu, os.path.join(sub_plots, "edge_mass_cdf_sparse.png"))

    # Top edges table for A (autonomous sparse)
    edges = topk_edges(W_sparse_cpu, k=args.top_edge_k)
    rows = []
    for (i, j, w) in edges:
        src_meta = rep_meta[j] if isinstance(rep_meta, list) and j < len(rep_meta) else None
        dst_meta = rep_meta[i] if isinstance(rep_meta, list) and i < len(rep_meta) else None
        rows.append([
            j, i, w, abs(w), 1 if w > 0 else (-1 if w < 0 else 0),
            json.dumps(src_meta, ensure_ascii=False) if src_meta is not None else "",
            json.dumps(dst_meta, ensure_ascii=False) if dst_meta is not None else "",
        ])

    _save_csv(os.path.join(sub_tables, "top_edges_autonomous_sparse.csv"),
              ["src_receptor_id", "dst_receptor_id", "weight", "abs_weight", "sign", "src_meta_json", "dst_meta_json"],
              rows)

    # Top edges table for B (DMDc selected model)
    # We'll output for B of chosen model tag.
    WB_cpu = WB_sparse.detach().cpu() if tag == "dmdc_sparse" else WB_ridge.detach().cpu()
    B_edges = topk_edges(WB_cpu.T, k=args.top_edge_k)  # WB: [C,C] mapping from U to Y; treat as edges into state
    rowsB = []
    for (i, j, w) in B_edges:
        src_meta = rep_meta[j] if isinstance(rep_meta, list) and j < len(rep_meta) else None
        dst_meta = rep_meta[i] if isinstance(rep_meta, list) and i < len(rep_meta) else None
        rowsB.append([
            j, i, w, abs(w), 1 if w > 0 else (-1 if w < 0 else 0),
            json.dumps(src_meta, ensure_ascii=False) if src_meta is not None else "",
            json.dumps(dst_meta, ensure_ascii=False) if dst_meta is not None else "",
        ])

    _save_csv(os.path.join(sub_tables, f"top_edges_B_{tag}.csv"),
              ["src_receptor_id", "dst_receptor_id", "weight", "abs_weight", "sign", "src_meta_json", "dst_meta_json"],
              rowsB)

    # Eigenvalue plots
    eig_auto = np.linalg.eigvals(_to_numpy(W_sparse_cpu))
    _try_plot_eigs(eig_auto, os.path.join(sub_plots, "eigvals_autonomous_sparse.png"))
    eig_dmdc = np.linalg.eigvals(_to_numpy(WA_sparse.detach().cpu()))
    _try_plot_eigs(eig_dmdc, os.path.join(sub_plots, "eigvals_dmdc_A_sparse.png"))

    np.save(os.path.join(sub_models, "eigs_autonomous_sparse.npy"), eig_auto)
    np.save(os.path.join(sub_models, "eigs_dmdc_A_sparse.npy"), eig_dmdc)

    # -------------------------
    # Success gates
    # -------------------------

    # G1: rollout beats identity by >=15%
    g1 = rollout_val["sparse"] <= 0.85 * rollout_val["identity"]
    # G2: DMDc improves one-step by >=5% over autonomous
    g2 = val_1["dmdc_sparse"] <= 0.95 * val_1["sparse"]
    # G3: after scaling, spectral radius <=1.05
    rho_ok = _spectral_radius_from_W(W_sparse_cpu) <= 1.05
    # G4: sparsity meaningful: density<=10%
    _, dens_sel = count_density(W_sparse_cpu, thr=args.edge_thr)
    g4 = dens_sel <= 0.10

    gates = {
        "G1_rollout_beats_identity_15pct": bool(g1),
        "G2_dmdc_beats_auto_5pct": bool(g2),
        "G3_spectral_radius_le_1p05": bool(rho_ok),
        "G4_density_le_0p10": bool(g4),
        "values": {
            "val_rollout_identity": float(rollout_val["identity"]),
            "val_rollout_sparse": float(rollout_val["sparse"]),
            "val_1step_sparse": float(val_1["sparse"]),
            "val_1step_dmdc_sparse": float(val_1["dmdc_sparse"]),
            "rho_sparse": float(_spectral_radius_from_W(W_sparse_cpu)),
            "density_sparse_report": float(density_report),
        },
    }

    _save_json(os.path.join(sub_metrics, "success_gate.json"), gates)

    # -------------------------
    # Write reports
    # -------------------------

    one_step = {
        "val": {k: float(v) for k, v in val_1.items()},
        "test": {k: float(v) for k, v in test_1.items()},
    }
    rollout = {
        "val": {k: float(v) for k, v in rollout_val.items()},
        "test": {k: float(v) for k, v in rollout_test.items()},
    }

    _save_json(os.path.join(sub_metrics, "one_step_metrics.json"), one_step)
    _save_json(os.path.join(sub_metrics, "rollout_metrics.json"), rollout)
    _save_json(os.path.join(sub_metrics, "control_sanity.json"), control_sanity)
    _save_json(os.path.join(sub_metrics, "graph_stats.json"), graph_stats)

    # Standardization save
    if std is not None:
        torch.save({
            "mu_x": std.mu_x.detach().cpu(),
            "sig_x": std.sig_x.detach().cpu(),
            "mu_u": std.mu_u.detach().cpu() if std.mu_u is not None else None,
            "sig_u": std.sig_u.detach().cpu() if std.sig_u is not None else None,
            "eps": float(args.eps),
        }, os.path.join(sub_models, "standardization.pt"))

    manifest = {
        "task": args.task,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "paths": {
            "fix3_dir": base_fix3,
            "out_e2": out_e2,
        },
        "shapes": {
            "C": int(C),
            "L": int(L),
            "N_train": int(N_train),
            "N_val": int(N_val),
            "N_test": int(N_test),
            "M": int(M),
        },
        "hyperparams": {
            "standardize": bool(args.standardize),
            "eps": float(args.eps),
            "ridge_lambdas": ridge_lams,
            "l1_lambdas": l1_lams,
            "fista_iters": int(args.fista_iters),
            "fista_tol": float(args.fista_tol),
            "rho_target": float(args.rho_target),
            "apply_stability_scaling": bool(args.apply_stability_scaling),
            "edge_thr": float(args.edge_thr),
            "density_thr": float(args.density_thr),
            "seed": int(args.seed),
            "shuffles": int(args.shuffles),
        },
    }
    _save_json(os.path.join(out_e2, "manifest.json"), manifest)

    final_report = {
        "C": int(C),
        "L": int(L),
        "N_train": int(N_train),
        "N_val": int(N_val),
        "N_test": int(N_test),
        "val_one_step": one_step["val"],
        "test_one_step": one_step["test"],
        "val_rollout": rollout["val"],
        "test_rollout": rollout["test"],
        "gates": gates,
        "control_sanity": control_sanity,
        "graph_stats": graph_stats,
        "notes": {
            "operator_convention": "rollout uses W = A.T; ridge/sparse fit W in Y≈XW",
        },
    }
    _save_json(os.path.join(sub_metrics, "final_report.json"), final_report)

    # Print compact summary
    print("=== E2 SUMMARY ===")
    print(f"C={C} L={L} N_train={N_train} N_val={N_val} N_test={N_test}")
    print("VAL one-step:", {k: round(v, 6) for k, v in one_step["val"].items()})
    print("TEST one-step:", {k: round(v, 6) for k, v in one_step["test"].items()})
    print("VAL rollout:", {k: round(v, 6) for k, v in rollout["val"].items()})
    print("TEST rollout:", {k: round(v, 6) for k, v in rollout["test"].items()})
    print("GATES:", {k: v for k, v in gates.items() if k.startswith("G")})
    print(f"[E2] wrote: {out_e2}")


if __name__ == "__main__":
    main()
