#!/usr/bin/env python3
"""
E2: Learn linear transport operators in receptor space.

Reads Fix3 receptor-state tensors:
  outputs/<task>/artifacts/e1_fix3/X_clean_{split}_fix3.pt  [N,L,C]
  outputs/<task>/artifacts/e1_fix3/X_corr_{split}_fix3.pt   [N,L,C]
and fits:
  - autonomous x_{t+1} ≈ A x_t
  - controlled  x_{t+1} ≈ A x_t + B u_t  where u_t = x_t^corr - x_t^clean

Saves models/metrics/plots/tables under:
  outputs/<task>/artifacts/e2_transport/
"""

from __future__ import annotations
import argparse, json, math, time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


# -------------------------
# Utilities
# -------------------------

def _now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()

def _load_pt(path: Path, map_location: str = "cpu"):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return torch.load(path, map_location=map_location)

def _save_json(path: Path, obj):
    _ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def _nrmse(Y: torch.Tensor, Yhat: torch.Tensor, eps: float = 1e-8) -> float:
    num = torch.linalg.norm(Y - Yhat, dim=1)
    den = torch.linalg.norm(Y, dim=1) + eps
    return float(torch.mean(num / den).item())

def _soft_threshold(W: torch.Tensor, thr: float) -> torch.Tensor:
    return torch.sign(W) * torch.clamp(torch.abs(W) - thr, min=0.0)

def _power_iteration_spectral_norm(X: torch.Tensor, n_iter: int = 20, eps: float = 1e-8) -> float:
    """
    Approximate ||X||_2 via power iteration on X^T X.
    X: [M, D]
    """
    _, D = X.shape
    v = torch.randn(D, device=X.device, dtype=X.dtype)
    v = v / (torch.linalg.norm(v) + eps)
    for _ in range(n_iter):
        v = X.T @ (X @ v)
        v = v / (torch.linalg.norm(v) + eps)
    Xv = X @ v
    norm = torch.linalg.norm(Xv) / (torch.linalg.norm(v) + eps)
    return float(norm.item())

def _spectral_radius(A: np.ndarray) -> float:
    eigvals = np.linalg.eigvals(A)
    return float(np.max(np.abs(eigvals)))

def _eigvals(A: np.ndarray) -> np.ndarray:
    return np.linalg.eigvals(A)

def _plot_eigs(eigs: np.ndarray, out_path: Path, title: str):
    plt.figure(figsize=(6, 6))
    plt.scatter(eigs.real, eigs.imag, s=10)
    theta = np.linspace(0, 2*np.pi, 512)
    plt.plot(np.cos(theta), np.sin(theta), linewidth=1)
    plt.axhline(0, linewidth=0.5)
    plt.axvline(0, linewidth=0.5)
    plt.title(title)
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def _plot_rollout_curve(curves: Dict[str, np.ndarray], out_path: Path, title: str):
    plt.figure(figsize=(7, 4))
    for name, arr in curves.items():
        plt.plot(np.arange(1, len(arr)+1), arr, label=name)
    plt.xlabel("Layer (1..L-1)")
    plt.ylabel("NRMSE")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def _plot_edge_mass_cdf(A: np.ndarray, out_path: Path, title: str):
    w = np.abs(A).ravel()
    w = w[w > 0]
    if w.size == 0:
        w = np.array([0.0])
    w_sorted = np.sort(w)[::-1]
    cdf = np.cumsum(w_sorted) / (np.sum(w_sorted) + 1e-12)
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, len(cdf)+1), cdf)
    plt.xlabel("Top-k edges")
    plt.ylabel("Cumulative |weight| mass fraction")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def _count_nnz(A: np.ndarray, edge_thr: float) -> int:
    return int(np.sum(np.abs(A) > edge_thr))

def _density(A: np.ndarray, edge_thr: float) -> float:
    C = A.shape[0]
    return _count_nnz(A, edge_thr) / float(C * C)

def _top_edge_mass_fracs(A: np.ndarray, ks=(10, 50, 200, 1000)) -> Dict[str, float]:
    w = np.abs(A).ravel()
    w_sorted = np.sort(w)[::-1]
    total = float(np.sum(w_sorted) + 1e-12)
    out = {}
    for k in ks:
        k = min(k, w_sorted.size)
        out[f"top{k}_edge_mass_frac"] = float(np.sum(w_sorted[:k]) / total)
    out["total_edge_mass"] = total
    return out

def _flatten_meta(meta):
    if not isinstance(meta, dict):
        return {}
    out = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple)):
            if len(v) <= 20 and all(isinstance(x, (str, int, float, bool)) for x in v):
                out[k] = ",".join(map(str, v))
            else:
                out[k] = f"<list:{len(v)}>"
        elif isinstance(v, dict):
            out[k] = f"<dict:{len(v)}>"
        else:
            out[k] = str(type(v))
    return out


# -------------------------
# Standardization operator conversions
# -------------------------

def std_to_orig_operator(A_std: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    Convert operator in standardized coordinates to operator on centered original coords:
      A_orig = diag(sigma) @ A_std @ diag(1/sigma)
    """
    sig = sigma
    inv = 1.0 / sig
    return (sig.view(-1, 1) * A_std) * inv.view(1, -1)

def std_to_orig_B(B_std: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return std_to_orig_operator(B_std, sigma)


# -------------------------
# Data construction
# -------------------------

@dataclass
class SplitData:
    X: torch.Tensor      # [M,C]
    Y: torch.Tensor      # [M,C]
    U: torch.Tensor      # [M,C]
    X_seq: torch.Tensor  # [N,L,C]
    U_seq: torch.Tensor  # [N,L,C]

def build_split_data(Xc: torch.Tensor, Xr: torch.Tensor) -> SplitData:
    if Xc.ndim != 3 or Xr.ndim != 3:
        raise ValueError(f"Expected [N,L,C]. Got Xc={tuple(Xc.shape)}, Xr={tuple(Xr.shape)}")
    if Xc.shape != Xr.shape:
        raise ValueError(f"Xc and Xr shape mismatch: {tuple(Xc.shape)} vs {tuple(Xr.shape)}")
    N, L, C = Xc.shape
    X = Xc[:, 0:L-1, :].reshape(N*(L-1), C)
    Y = Xc[:, 1:L,   :].reshape(N*(L-1), C)
    U = (Xr - Xc)[:, 0:L-1, :].reshape(N*(L-1), C)
    return SplitData(X=X, Y=Y, U=U, X_seq=Xc, U_seq=(Xr - Xc))

@dataclass
class Standardizer:
    mu: torch.Tensor
    sigma: torch.Tensor
    eps: float = 1e-8
    def transform_X(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.mu) / self.sigma
    def transform_Y(self, Y: torch.Tensor) -> torch.Tensor:
        return (Y - self.mu) / self.sigma
    def transform_U(self, U: torch.Tensor) -> torch.Tensor:
        return U / self.sigma
    def to(self, device: torch.device, dtype: torch.dtype) -> "Standardizer":
        return Standardizer(mu=self.mu.to(device=device, dtype=dtype),
                            sigma=self.sigma.to(device=device, dtype=dtype),
                            eps=self.eps)

def fit_standardizer(X_train: torch.Tensor, eps: float = 1e-8) -> Standardizer:
    mu = torch.mean(X_train, dim=0)
    sigma = torch.std(X_train, dim=0, unbiased=False) + eps
    return Standardizer(mu=mu, sigma=sigma, eps=eps)


# -------------------------
# Model fitting
# -------------------------

def fit_diag(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    num = torch.sum(X * Y, dim=0)
    den = torch.sum(X * X, dim=0) + eps
    a = num / den
    return torch.diag(a)

def fit_ridge_autonomous(X: torch.Tensor, Y: torch.Tensor, lam: float) -> torch.Tensor:
    C = X.shape[1]
    G = (X.T @ X) + lam * torch.eye(C, device=X.device, dtype=X.dtype)
    RHS = Y.T @ X
    A = torch.linalg.solve(G.T, RHS.T).T
    return A

def fit_ridge_dmdc(X: torch.Tensor, U: torch.Tensor, Y: torch.Tensor, lam: float) -> Tuple[torch.Tensor, torch.Tensor]:
    C = X.shape[1]
    Z = torch.cat([X, U], dim=1)
    D = Z.shape[1]
    G = (Z.T @ Z) + lam * torch.eye(D, device=Z.device, dtype=Z.dtype)
    RHS = Y.T @ Z
    Theta = torch.linalg.solve(G.T, RHS.T).T
    return Theta[:, :C], Theta[:, C:]

def _objective(X: torch.Tensor, Y: torch.Tensor, W: torch.Tensor, l1: float) -> float:
    Yhat = X @ W.T
    loss = 0.5 * torch.mean((Yhat - Y) ** 2)
    reg = l1 * torch.mean(torch.abs(W))
    return float((loss + reg).item())

@dataclass
class FistaResult:
    W: torch.Tensor
    train_obj: float
    val_nrmse: float
    n_iter: int

def fit_fista_l1(
    X_train: torch.Tensor, Y_train: torch.Tensor,
    X_val: torch.Tensor,   Y_val: torch.Tensor,
    l1_lambda: float,
    W0: torch.Tensor,
    max_iters: int = 2000,
    tol: float = 1e-6,
    eval_every: int = 50,
    patience: int = 10,
    power_iters: int = 15,
    eps: float = 1e-8,
) -> FistaResult:
    M = X_train.shape[0]
    spec = _power_iteration_spectral_norm(X_train, n_iter=power_iters, eps=eps)
    L = max(spec * spec, eps)
    eta = 1.0 / (L + eps)

    W = W0.clone()
    Z = W.clone()
    t = 1.0

    best_val = float("inf")
    best_W = W.clone()
    best_iter = 0
    no_improve = 0
    last_obj = None

    for it in range(1, max_iters + 1):
        Yhat = X_train @ Z.T
        E = (Yhat - Y_train)
        grad = (E.T @ X_train) / float(M)

        W_next = _soft_threshold(Z - eta * grad, eta * l1_lambda)
        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        Z = W_next + ((t - 1.0) / t_next) * (W_next - W)
        W = W_next
        t = t_next

        if it % eval_every == 0 or it == 1 or it == max_iters:
            with torch.no_grad():
                val_nrmse = _nrmse(Y_val, X_val @ W.T, eps=eps)

            if val_nrmse + 1e-12 < best_val:
                best_val = val_nrmse
                best_W = W.clone()
                best_iter = it
                no_improve = 0
            else:
                no_improve += 1

            obj = _objective(X_train, Y_train, W, l1_lambda)
            if last_obj is not None:
                rel = abs(last_obj - obj) / (abs(last_obj) + eps)
                if rel < tol:
                    no_improve += 1
            last_obj = obj

            if no_improve >= patience:
                break

    final_obj = _objective(X_train, Y_train, best_W, l1_lambda)
    return FistaResult(W=best_W, train_obj=final_obj, val_nrmse=float(best_val), n_iter=int(best_iter))


# -------------------------
# Evaluation: rollout
# -------------------------

def rollout_nrmse_by_layer(X_seq: torch.Tensor, A: torch.Tensor, eps: float = 1e-8) -> Tuple[float, np.ndarray]:
    N, L, C = X_seq.shape
    xhat = X_seq[:, 0, :].clone()
    errs = []
    for t in range(L - 1):
        xhat = xhat @ A.T
        xt = X_seq[:, t + 1, :]
        num = torch.linalg.norm(xt - xhat, dim=1)
        den = torch.linalg.norm(xt, dim=1) + eps
        errs.append(torch.mean(num / den).item())
    errs_arr = np.array(errs, dtype=np.float64)
    return float(np.mean(errs_arr)), errs_arr


# -------------------------
# Top edges table
# -------------------------

def write_top_edges_csv(W: np.ndarray, out_csv: Path, rep_meta: Optional[list],
                        top_k: int = 200, edge_thr: float = 1e-4,
                        src_offset: int = 0, src_prefix: str = "src", dst_prefix: str = "dst"):
    C, D = W.shape
    absW = np.abs(W)
    mask = absW > edge_thr
    idxs = np.argwhere(mask)
    if idxs.size == 0:
        df = pd.DataFrame(columns=[f"{src_prefix}_id", f"{dst_prefix}_id", "weight", "abs_weight", "sign"])
        _ensure_dir(out_csv.parent)
        df.to_csv(out_csv, index=False)
        return

    vals = absW[mask]
    order = np.argsort(vals)[::-1][: min(top_k, vals.size)]
    top_pairs = idxs[order]
    top_abs = vals[order]
    top_w = W[top_pairs[:, 0], top_pairs[:, 1]]

    rows = []
    for (i, j), w, aw in zip(top_pairs, top_w, top_abs):
        row = {
            f"{dst_prefix}_id": int(i),
            f"{src_prefix}_id": int(j + src_offset),
            "weight": float(w),
            "abs_weight": float(aw),
            "sign": int(np.sign(w)) if w != 0 else 0,
        }
        if rep_meta is not None and j < len(rep_meta):
            sm = _flatten_meta(rep_meta[j])
            for k, v in sm.items():
                row[f"{src_prefix}_{k}"] = v
        if rep_meta is not None and i < len(rep_meta):
            dm = _flatten_meta(rep_meta[i])
            for k, v in dm.items():
                row[f"{dst_prefix}_{k}"] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    _ensure_dir(out_csv.parent)
    df.to_csv(out_csv, index=False)


# -------------------------
# Main
# -------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, required=True, choices=["gp", "ioi", "gt"])
    p.add_argument("--out_dir", type=str, default=None, help="Path like outputs/gp. If omitted, uses --out_root/<task>.")
    p.add_argument("--out_root", type=str, default=None, help="Alternative to --out_dir: root like outputs")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])
    p.add_argument("--eps", type=float, default=1e-8)

    p.add_argument("--standardize", action="store_true", default=True)
    p.add_argument("--no_standardize", action="store_true", default=False)

    p.add_argument("--ridge_lambdas", type=str, default="1e-4,1e-3,1e-2,1e-1")
    p.add_argument("--l1_lambdas", type=str, default="0,1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3")
    p.add_argument("--fista_iters", type=int, default=2000)
    p.add_argument("--fista_tol", type=float, default=1e-6)
    p.add_argument("--fista_eval_every", type=int, default=50)
    p.add_argument("--fista_patience", type=int, default=10)

    p.add_argument("--rho_target", type=float, default=0.99)
    p.add_argument("--apply_stability_scaling", action="store_true", default=True)
    p.add_argument("--no_stability_scaling", action="store_true", default=False)

    p.add_argument("--edge_thr", type=float, default=1e-4)
    p.add_argument("--top_edge_k", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.no_standardize:
        args.standardize = False
    if args.no_stability_scaling:
        args.apply_stability_scaling = False

    if args.out_dir is None:
        if args.out_root is None:
            raise ValueError("Provide --out_dir or --out_root")
        out_dir = Path(args.out_root) / args.task
    else:
        out_dir = Path(args.out_dir)

    base_e1 = out_dir / "artifacts" / "e1_fix3"
    bank_path = base_e1 / "receptor_bank.pt"
    Xc_train_p = base_e1 / "X_clean_train_fix3.pt"
    Xc_val_p   = base_e1 / "X_clean_val_fix3.pt"
    Xc_test_p  = base_e1 / "X_clean_test_fix3.pt"
    Xr_train_p = base_e1 / "X_corr_train_fix3.pt"
    Xr_val_p   = base_e1 / "X_corr_val_fix3.pt"
    Xr_test_p  = base_e1 / "X_corr_test_fix3.pt"

    out_e2 = out_dir / "artifacts" / "e2_transport"
    models_dir = _ensure_dir(out_e2 / "models")
    metrics_dir = _ensure_dir(out_e2 / "metrics")
    plots_dir = _ensure_dir(out_e2 / "plots")
    tables_dir = _ensure_dir(out_e2 / "tables")
    debug_dir = _ensure_dir(out_e2 / "debug")

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device)
    dtype = torch.float32

    print(f"[E2] task={args.task} out_dir={out_dir} device={device} dtype={dtype}")
    print(f"[E2] reading Fix3 artifacts from: {base_e1}")

    bank = _load_pt(bank_path, map_location="cpu")
    rep_meta = bank.get("rep_meta", None)
    v_bank = bank.get("v_bank", bank.get("v_c", None))
    if v_bank is None:
        raise KeyError(f"receptor_bank.pt missing v_bank/v_c keys: {list(bank.keys())}")

    Xc_train = _load_pt(Xc_train_p, map_location="cpu")
    Xc_val   = _load_pt(Xc_val_p,   map_location="cpu")
    Xc_test  = _load_pt(Xc_test_p,  map_location="cpu")
    Xr_train = _load_pt(Xr_train_p, map_location="cpu")
    Xr_val   = _load_pt(Xr_val_p,   map_location="cpu")
    Xr_test  = _load_pt(Xr_test_p,  map_location="cpu")

    N_train, L, C = Xc_train.shape
    N_val, _, _ = Xc_val.shape
    N_test, _, _ = Xc_test.shape
    assert len(rep_meta) == C, f"rep_meta length mismatch: {len(rep_meta)} vs C={C}"
    print(f"[E2] Shapes: train={tuple(Xc_train.shape)} val={tuple(Xc_val.shape)} test={tuple(Xc_test.shape)}")

    train = build_split_data(Xc_train, Xr_train)
    val   = build_split_data(Xc_val,   Xr_val)
    test  = build_split_data(Xc_test,  Xr_test)

    train_X = train.X.to(device=device, dtype=dtype)
    train_Y = train.Y.to(device=device, dtype=dtype)
    train_U = train.U.to(device=device, dtype=dtype)

    val_X   = val.X.to(device=device, dtype=dtype)
    val_Y   = val.Y.to(device=device, dtype=dtype)
    val_U   = val.U.to(device=device, dtype=dtype)

    test_X  = test.X.to(device=device, dtype=dtype)
    test_Y  = test.Y.to(device=device, dtype=dtype)
    test_U  = test.U.to(device=device, dtype=dtype)

    eps = args.eps
    if args.standardize:
        std = fit_standardizer(train_X, eps=eps).to(device=device, dtype=dtype)
        Xtr = std.transform_X(train_X); Ytr = std.transform_Y(train_Y); Utr = std.transform_U(train_U)
        Xva = std.transform_X(val_X);   Yva = std.transform_Y(val_Y);   Uva = std.transform_U(val_U)
        Xte = std.transform_X(test_X);  Yte = std.transform_Y(test_Y);  Ute = std.transform_U(test_U)
        torch.save({"mu": std.mu.detach().cpu(), "sigma": std.sigma.detach().cpu(), "eps": eps}, models_dir / "standardization.pt")
    else:
        std = None
        Xtr, Ytr, Utr = train_X, train_Y, train_U
        Xva, Yva, Uva = val_X, val_Y, val_U
        Xte, Yte, Ute = test_X, test_Y, test_U

    A_id = torch.eye(C, device=device, dtype=dtype)
    A_diag = fit_diag(Xtr, Ytr, eps=eps).to(device=device)

    ridge_lams = [float(x) for x in args.ridge_lambdas.split(",") if x.strip() != ""]
    l1_lams = [float(x) for x in args.l1_lambdas.split(",") if x.strip() != ""]

    def eval_auto(A: torch.Tensor, X: torch.Tensor, Y: torch.Tensor) -> float:
        return _nrmse(Y, X @ A.T, eps=eps)

    def eval_dmdc(A: torch.Tensor, B: torch.Tensor, X: torch.Tensor, U: torch.Tensor, Y: torch.Tensor) -> float:
        return _nrmse(Y, X @ A.T + U @ B.T, eps=eps)

    # Ridge autonomous
    ridge_rows = []
    best_ridge = None
    best_ridge_val = float("inf")
    best_ridge_lam = None
    for lam in ridge_lams:
        A = fit_ridge_autonomous(Xtr, Ytr, lam=lam)
        v = eval_auto(A, Xva, Yva)
        ridge_rows.append({"lambda": lam, "val_nrmse_1step": v})
        if v < best_ridge_val:
            best_ridge_val, best_ridge, best_ridge_lam = v, A, lam
    pd.DataFrame(ridge_rows).to_csv(tables_dir / "ridge_sweep_autonomous.csv", index=False)

    # Ridge DMDc
    ridgec_rows = []
    best_ridgec = None
    best_ridgec_val = float("inf")
    best_ridgec_lam = None
    for lam in ridge_lams:
        A, B = fit_ridge_dmdc(Xtr, Utr, Ytr, lam=lam)
        v = eval_dmdc(A, B, Xva, Uva, Yva)
        ridgec_rows.append({"lambda": lam, "val_nrmse_1step": v})
        if v < best_ridgec_val:
            best_ridgec_val, best_ridgec, best_ridgec_lam = v, (A, B), lam
    pd.DataFrame(ridgec_rows).to_csv(tables_dir / "ridge_sweep_dmdc.csv", index=False)

    # Sparse autonomous: lambda sweep
    warm_A = best_ridge.detach()
    sweep = []
    for l1 in l1_lams:
        res = fit_fista_l1(Xtr, Ytr, Xva, Yva, l1_lambda=l1, W0=warm_A,
                           max_iters=args.fista_iters, tol=args.fista_tol,
                           eval_every=args.fista_eval_every, patience=args.fista_patience, eps=eps)
        A = res.W
        dens = _density(_to_numpy(A), edge_thr=args.edge_thr)
        sweep.append({"l1_lambda": l1, "val_nrmse_1step": res.val_nrmse, "train_obj": res.train_obj,
                      "n_iter": res.n_iter, "density": dens})
    df_sweep = pd.DataFrame(sweep)
    df_sweep.to_csv(tables_dir / "lambda_sweep_autonomous.csv", index=False)

    df_ok = df_sweep[df_sweep["density"] <= 0.10]
    pick = df_ok.sort_values("val_nrmse_1step").iloc[0] if len(df_ok) else df_sweep.sort_values("val_nrmse_1step").iloc[0]
    best_sparse_l1 = float(pick["l1_lambda"])
    res = fit_fista_l1(Xtr, Ytr, Xva, Yva, l1_lambda=best_sparse_l1, W0=warm_A,
                       max_iters=args.fista_iters, tol=args.fista_tol,
                       eval_every=args.fista_eval_every, patience=args.fista_patience, eps=eps)
    A_sparse = res.W

    # Sparse DMDc (Theta)
    A_ridgec_best, B_ridgec_best = best_ridgec
    warm_Theta = torch.cat([A_ridgec_best, B_ridgec_best], dim=1).detach()
    Ztr = torch.cat([Xtr, Utr], dim=1)
    Zva = torch.cat([Xva, Uva], dim=1)

    sweepc = []
    for l1 in l1_lams:
        res = fit_fista_l1(Ztr, Ytr, Zva, Yva, l1_lambda=l1, W0=warm_Theta,
                           max_iters=args.fista_iters, tol=args.fista_tol,
                           eval_every=args.fista_eval_every, patience=args.fista_patience, eps=eps)
        Theta = res.W
        densA = _density(_to_numpy(Theta[:, :C]), edge_thr=args.edge_thr)
        sweepc.append({"l1_lambda": l1, "val_nrmse_1step": res.val_nrmse, "train_obj": res.train_obj,
                       "n_iter": res.n_iter, "density_A": densA})
    df_sweepc = pd.DataFrame(sweepc)
    df_sweepc.to_csv(tables_dir / "lambda_sweep_dmdc.csv", index=False)

    df_ok = df_sweepc[df_sweepc["density_A"] <= 0.10]
    pick = df_ok.sort_values("val_nrmse_1step").iloc[0] if len(df_ok) else df_sweepc.sort_values("val_nrmse_1step").iloc[0]
    best_sparsec_l1 = float(pick["l1_lambda"])
    res = fit_fista_l1(Ztr, Ytr, Zva, Yva, l1_lambda=best_sparsec_l1, W0=warm_Theta,
                       max_iters=args.fista_iters, tol=args.fista_tol,
                       eval_every=args.fista_eval_every, patience=args.fista_patience, eps=eps)
    Theta_best = res.W
    A_dmdc = Theta_best[:, :C]
    B_dmdc = Theta_best[:, C:]

    # Stability scaling for A blocks
    def stabilize(A: torch.Tensor, name: str) -> Tuple[torch.Tensor, Dict]:
        rho_before = _spectral_radius(_to_numpy(A))
        A2 = A
        scale = 1.0
        rho_after = rho_before
        if args.apply_stability_scaling and rho_before > args.rho_target:
            scale = args.rho_target / max(rho_before, 1e-12)
            A2 = A * scale
            rho_after = _spectral_radius(_to_numpy(A2))
        return A2, {"name": name, "rho_before": rho_before, "rho_after": rho_after, "scale": scale}

    A_ridge_best = best_ridge
    A_ridge_best, stab_ridge = stabilize(A_ridge_best, "A_ridge")
    A_sparse, stab_sparse = stabilize(A_sparse, "A_sparse")
    A_dmdc, stab_dmdc = stabilize(A_dmdc, "A_dmdc_A")

    _save_json(metrics_dir / "stability.json", {
        "ridge": stab_ridge, "sparse": stab_sparse, "dmdc_sparse_A": stab_dmdc,
        "rho_target": args.rho_target, "apply_stability_scaling": args.apply_stability_scaling
    })

    # Eigs + plots
    eig_auto = _eigvals(_to_numpy(A_sparse))
    eig_dmdc = _eigvals(_to_numpy(A_dmdc))
    np.save(models_dir / "eigs_autonomous.npy", eig_auto)
    np.save(models_dir / "eigs_dmdc.npy", eig_dmdc)
    _plot_eigs(eig_auto, plots_dir / "eigvals_autonomous_sparse.png", "Eigenvalues: Autonomous sparse A (std)")
    _plot_eigs(eig_dmdc, plots_dir / "eigvals_dmdc_sparse_A.png", "Eigenvalues: DMDc sparse A block (std)")

    # One-step metrics
    models_auto = {"identity": A_id, "diag": A_diag, "ridge": A_ridge_best, "sparse": A_sparse}
    models_dmdc = {"dmdc_ridge": (A_ridgec_best, B_ridgec_best), "dmdc_sparse": (A_dmdc, B_dmdc)}
    one_step = {"val": {}, "test": {}}
    for split_name, Xs, Ys, Us in [("val", Xva, Yva, Uva), ("test", Xte, Yte, Ute)]:
        for nm, A in models_auto.items():
            one_step[split_name][nm] = _nrmse(Ys, Xs @ A.T, eps=eps)
        for nm, (A, B) in models_dmdc.items():
            one_step[split_name][nm] = _nrmse(Ys, Xs @ A.T + Us @ B.T, eps=eps)
    _save_json(metrics_dir / "one_step_metrics.json", one_step)

    # Control sanity: shuffle U rows in train, refit DMDc sparse at best l1, evaluate on true U
    perm = torch.randperm(Utr.shape[0], device=Utr.device)
    Utr_shuf = Utr[perm]
    Ztr_shuf = torch.cat([Xtr, Utr_shuf], dim=1)
    res_shuf = fit_fista_l1(Ztr_shuf, Ytr, Zva, Yva, l1_lambda=best_sparsec_l1, W0=warm_Theta,
                            max_iters=args.fista_iters, tol=args.fista_tol,
                            eval_every=args.fista_eval_every, patience=args.fista_patience, eps=eps)
    Theta_shuf = res_shuf.W
    A_shuf = Theta_shuf[:, :C]
    B_shuf = Theta_shuf[:, C:]
    shuf_val = _nrmse(Yva, Xva @ A_shuf.T + Uva @ B_shuf.T, eps=eps)
    shuf_test = _nrmse(Yte, Xte @ A_shuf.T + Ute @ B_shuf.T, eps=eps)

    ctrl_sanity = {
        "best_l1_lambda": best_sparsec_l1,
        "val_nrmse_dmdc_sparse": one_step["val"]["dmdc_sparse"],
        "val_nrmse_shuffled_control": shuf_val,
        "test_nrmse_dmdc_sparse": one_step["test"]["dmdc_sparse"],
        "test_nrmse_shuffled_control": shuf_test,
        "drop_frac_val": float((shuf_val - one_step["val"]["dmdc_sparse"]) / max(one_step["val"]["dmdc_sparse"], eps)),
        "drop_frac_test": float((shuf_test - one_step["test"]["dmdc_sparse"]) / max(one_step["test"]["dmdc_sparse"], eps)),
        "mean_control_norm_train": float(torch.mean(torch.linalg.norm(Utr, dim=1)).item()),
        "mean_control_norm_val": float(torch.mean(torch.linalg.norm(Uva, dim=1)).item()),
    }
    _save_json(metrics_dir / "control_sanity.json", ctrl_sanity)

    # Rollout (autonomous)
    Xc_val_seq = val.X_seq.to(device=device, dtype=dtype)
    Xc_test_seq = test.X_seq.to(device=device, dtype=dtype)
    if args.standardize:
        mu = std.mu.view(1, 1, C)
        sig = std.sigma.view(1, 1, C)
        Xc_val_seq = (Xc_val_seq - mu) / sig
        Xc_test_seq = (Xc_test_seq - mu) / sig

    rollout = {"val": {}, "test": {}}
    rollout_by_layer = {"val": {}, "test": {}}
    for split_name, Xseq in [("val", Xc_val_seq), ("test", Xc_test_seq)]:
        for nm, A in models_auto.items():
            total, by_layer = rollout_nrmse_by_layer(Xseq, A, eps=eps)
            rollout[split_name][nm] = total
            rollout_by_layer[split_name][nm] = by_layer.tolist()
    _save_json(metrics_dir / "rollout_metrics.json", rollout)

    np.save(debug_dir / "rollout_err_by_layer.npy", np.array(rollout_by_layer["val"]["sparse"], dtype=np.float32))
    np.save(debug_dir / "rollout_err_by_layer_val.npy", np.array(rollout_by_layer["val"]["sparse"], dtype=np.float32))
    np.save(debug_dir / "rollout_err_by_layer_test.npy", np.array(rollout_by_layer["test"]["sparse"], dtype=np.float32))

    _plot_rollout_curve({k: np.array(rollout_by_layer["val"][k]) for k in ["identity","diag","ridge","sparse"]},
                        plots_dir / "rollout_error_curve_val.png", "Rollout error by layer (VAL)")
    _plot_rollout_curve({k: np.array(rollout_by_layer["test"][k]) for k in ["identity","diag","ridge","sparse"]},
                        plots_dir / "rollout_error_curve_test.png", "Rollout error by layer (TEST)")

    # Interpretability stats + plots
    A_sparse_np = _to_numpy(A_sparse)
    A_dmdc_np = _to_numpy(A_dmdc)
    B_dmdc_np = _to_numpy(B_dmdc)
    graph_stats = {
        "autonomous_sparse": {"density": _density(A_sparse_np, args.edge_thr), "nnz": _count_nnz(A_sparse_np, args.edge_thr),
                              **_top_edge_mass_fracs(A_sparse_np)},
        "dmdc_sparse_A": {"density": _density(A_dmdc_np, args.edge_thr), "nnz": _count_nnz(A_dmdc_np, args.edge_thr),
                          **_top_edge_mass_fracs(A_dmdc_np)},
        "dmdc_sparse_B": {"density": float(_count_nnz(B_dmdc_np, args.edge_thr) / (C*C)), "nnz": _count_nnz(B_dmdc_np, args.edge_thr),
                          **_top_edge_mass_fracs(B_dmdc_np)}
    }
    _save_json(metrics_dir / "graph_stats.json", graph_stats)

    _plot_edge_mass_cdf(A_sparse_np, plots_dir / "edge_mass_cdf_autonomous_sparse.png", "Edge mass CDF: A (auto sparse)")
    _plot_edge_mass_cdf(A_dmdc_np, plots_dir / "edge_mass_cdf_dmdc_A.png", "Edge mass CDF: A (dmdc sparse)")
    _plot_edge_mass_cdf(B_dmdc_np, plots_dir / "edge_mass_cdf_dmdc_B.png", "Edge mass CDF: B (dmdc sparse)")

    write_top_edges_csv(A_sparse_np, tables_dir / "top_edges_autonomous.csv", rep_meta, top_k=args.top_edge_k, edge_thr=args.edge_thr)
    write_top_edges_csv(A_dmdc_np, tables_dir / "top_edges_dmdc_A.csv", rep_meta, top_k=args.top_edge_k, edge_thr=args.edge_thr)
    write_top_edges_csv(B_dmdc_np, tables_dir / "top_edges_B.csv", rep_meta, top_k=args.top_edge_k, edge_thr=args.edge_thr)

    # Save models (std + orig)
    if args.standardize:
        sigma_cpu = std.sigma.detach().cpu()
        A_ridge_orig = std_to_orig_operator(A_ridge_best.detach().cpu(), sigma_cpu)
        A_sparse_orig = std_to_orig_operator(A_sparse.detach().cpu(), sigma_cpu)
        A_dmdc_orig = std_to_orig_operator(A_dmdc.detach().cpu(), sigma_cpu)
        B_dmdc_orig = std_to_orig_B(B_dmdc.detach().cpu(), sigma_cpu)
    else:
        A_ridge_orig = A_ridge_best.detach().cpu()
        A_sparse_orig = A_sparse.detach().cpu()
        A_dmdc_orig = A_dmdc.detach().cpu()
        B_dmdc_orig = B_dmdc.detach().cpu()

    torch.save({"A_std": A_ridge_best.detach().cpu(), "A_orig": A_ridge_orig, "lambda": best_ridge_lam},
               models_dir / "A_autonomous_ridge.pt")
    torch.save({"A_std": A_sparse.detach().cpu(), "A_orig": A_sparse_orig, "l1_lambda": best_sparse_l1},
               models_dir / "A_autonomous_sparse.pt")
    torch.save({"A_std": A_dmdc.detach().cpu(), "A_orig": A_dmdc_orig, "l1_lambda": best_sparsec_l1},
               models_dir / "A_dmdc_sparse.pt")
    torch.save({"B_std": B_dmdc.detach().cpu(), "B_orig": B_dmdc_orig, "l1_lambda": best_sparsec_l1},
               models_dir / "B_dmdc_sparse.pt")

    # Gates
    g1 = rollout["val"]["sparse"] <= 0.85 * rollout["val"]["identity"]
    g2 = one_step["val"]["dmdc_sparse"] <= 0.95 * one_step["val"]["sparse"]
    g3 = stab_sparse["rho_after"] <= 1.05
    g4 = graph_stats["autonomous_sparse"]["density"] <= 0.10

    success_gate = {
        "G1_rollout_beats_identity_15pct": bool(g1),
        "G2_dmdc_beats_auto_5pct": bool(g2),
        "G3_spectral_radius_le_1p05": bool(g3),
        "G4_density_le_0p10": bool(g4),
        "values": {
            "val_rollout_identity": rollout["val"]["identity"],
            "val_rollout_sparse": rollout["val"]["sparse"],
            "val_1step_sparse": one_step["val"]["sparse"],
            "val_1step_dmdc_sparse": one_step["val"]["dmdc_sparse"],
            "rho_sparse_after": stab_sparse["rho_after"],
            "density_sparse": graph_stats["autonomous_sparse"]["density"],
        }
    }
    _save_json(metrics_dir / "success_gate.json", success_gate)

    manifest = {
        "task": args.task,
        "timestamp": _now_ts(),
        "paths": {"bank": str(bank_path),
                  "X_clean_train": str(Xc_train_p), "X_clean_val": str(Xc_val_p), "X_clean_test": str(Xc_test_p),
                  "X_corr_train": str(Xr_train_p), "X_corr_val": str(Xr_val_p), "X_corr_test": str(Xr_test_p)},
        "sizes": {"C": C, "L": L, "N_train": N_train, "N_val": N_val, "N_test": N_test},
        "hyperparams": {"standardize": args.standardize, "eps": eps, "ridge_lambdas": ridge_lams, "l1_lambdas": l1_lams,
                        "fista_iters": args.fista_iters, "fista_tol": args.fista_tol,
                        "fista_eval_every": args.fista_eval_every, "fista_patience": args.fista_patience,
                        "rho_target": args.rho_target, "apply_stability_scaling": args.apply_stability_scaling,
                        "edge_thr": args.edge_thr, "top_edge_k": args.top_edge_k,
                        "device": str(device), "dtype": str(dtype), "seed": args.seed,
                        "best_lambdas": {"ridge_autonomous": best_ridge_lam, "ridge_dmdc": best_ridgec_lam,
                                         "sparse_autonomous_l1": best_sparse_l1, "sparse_dmdc_l1": best_sparsec_l1}}
    }
    _save_json(out_e2 / "manifest.json", manifest)

    final_report = {
        "manifest": manifest,
        "one_step_metrics": one_step,
        "rollout_metrics": rollout,
        "stability": {"ridge": stab_ridge, "sparse": stab_sparse, "dmdc_sparse_A": stab_dmdc},
        "control_sanity": ctrl_sanity,
        "graph_stats": graph_stats,
        "success_gate": success_gate,
    }
    _save_json(metrics_dir / "final_report.json", final_report)

    print("\n=== E2 SUMMARY ===")
    print(f"C={C} L={L} N_train={N_train} N_val={N_val} N_test={N_test}")
    print("VAL one-step:", {k: round(v, 4) for k, v in one_step["val"].items()})
    print("TEST one-step:", {k: round(v, 4) for k, v in one_step["test"].items()})
    print("VAL rollout:", {k: round(v, 4) for k, v in rollout["val"].items()})
    print("TEST rollout:", {k: round(v, 4) for k, v in rollout["test"].items()})
    print("GATES:", {k: v for k, v in success_gate.items() if k != "values"})
    print(f"[E2] wrote: {out_e2}")

if __name__ == "__main__":
    main()
