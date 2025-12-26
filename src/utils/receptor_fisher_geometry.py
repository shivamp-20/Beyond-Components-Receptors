"""
Fisher/KL geometry utilities for logit receptors.

Matches Beyond Components Appendix B.2 framing:
- write directions are RIGHT singular vectors in residual space (columns of V_write)
- logit receptors live in vocab-logit space after projection through unembedding W_U

This file is intentionally dependency-light (only torch + stdlib) and does not mutate models.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple, Union

import torch
import torch.nn.functional as F


@torch.no_grad()
def make_logit_receptors(V_write: torch.Tensor, W_U_raw: torch.Tensor) -> torch.Tensor:
    """
    Build logit-space receptor matrix from residual-space write directions.

    Paper form:
        receptor_logits = v^T W_U   (with W_U shaped [d_model, vocab])

    In code we return:
        R_logit: [vocab, m]  (one column per receptor)

    Args:
        V_write: [d_model, m]
        W_U_raw: raw unembedding weight, either [d_model, vocab] or [vocab, d_model]

    Returns:
        R_logit: [vocab, m]
    """
    if V_write.ndim != 2:
        raise ValueError(f"V_write must be rank-2 [d_model, m]; got {tuple(V_write.shape)}")
    if W_U_raw.ndim != 2:
        raise ValueError(f"W_U_raw must be rank-2; got {tuple(W_U_raw.shape)}")

    d_model = int(V_write.shape[0])
    w0, w1 = map(int, W_U_raw.shape)

    # Convert to "paper form" W_U_paper: [d_model, vocab]
    if w0 == d_model:
        W_U_paper = W_U_raw
        _vocab = w1
    elif w1 == d_model:
        W_U_paper = W_U_raw.T
        _vocab = w0
    else:
        raise ValueError(
            f"Unembed weight shape {tuple(W_U_raw.shape)} does not match V_write d_model={d_model}. "
            "Expected W_U_raw to be [d_model, vocab] or [vocab, d_model]."
        )

    # R_rows: [m, vocab] then transpose to [vocab, m]
    R_rows = V_write.T @ W_U_paper
    R_logit = R_rows.T

    # Return float32 unless any input is float64
    if V_write.dtype == torch.float64 or W_U_raw.dtype == torch.float64:
        return R_logit.to(dtype=torch.float64)
    return R_logit.to(dtype=torch.float32)


def _extract_forward_model(model: Any) -> Any:
    """
    Support either a callable model that returns logits, or a wrapper with `.model`.
    """
    if callable(model):
        return model
    if hasattr(model, "model") and callable(getattr(model, "model")):
        return getattr(model, "model")
    raise TypeError("model must be callable (returns logits) or have a callable `.model` attribute.")


@torch.no_grad()
def fisher_gram_streaming(
    R_logit: torch.Tensor,
    model: Any,
    dataloader: Iterable,
    device: Union[str, torch.device],
    position: Union[str, int] = "last",
    max_batches: int = 50,
    return_N: bool = True,
    dtype: torch.dtype = torch.float64,
) -> Union[torch.Tensor, Tuple[torch.Tensor, int]]:
    """
    Compute Fisher/KL Gram matrix:
        G = E_x [ R^T F(p(x)) R ], with F(p)=diag(p) - p p^T, p=softmax(logits)

    Efficient identity (never materialize F):
        G = R^T diag(p_bar) R  -  E[ a a^T ]
    where:
        p_bar = E[p]
        a = R^T p  (equivalently a = p @ R)

    Streaming accumulation (no storing all p):
        p_sum[vocab] += sum_b p_b
        B_sum[m,m]  += A^T A   where A = p @ R

    The dataloader is expected to yield tokenized batches, either:
        - dict with 'input_ids' and optional 'attention_mask'
        - tuple/list (input_ids, attention_mask)  (attention_mask may be None)
    """
    if R_logit.ndim != 2:
        raise ValueError(f"R_logit must be [vocab, m]; got {tuple(R_logit.shape)}")

    forward_model = _extract_forward_model(model)
    device = torch.device(device)

    vocab, m = map(int, R_logit.shape)
    R_logit = R_logit.to(device=device)

    p_sum = torch.zeros(vocab, device=device, dtype=dtype)
    B_sum = torch.zeros(m, m, device=device, dtype=dtype)
    N = 0

    # Try to infer pad_token_id if available (HF/TransformerLens tokenizers)
    pad_token_id = None
    tok = getattr(model, "tokenizer", None) or getattr(getattr(model, "model", None), "tokenizer", None)
    if tok is not None:
        pad_token_id = getattr(tok, "pad_token_id", None)

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break

        if isinstance(batch, (tuple, list)):
            input_ids = batch[0]
            attention_mask = batch[1] if len(batch) > 1 else None
        elif isinstance(batch, dict):
            if "input_ids" not in batch:
                raise KeyError(
                    "fisher_gram_streaming expects tokenized batches. "
                    "Batch dict must include key 'input_ids'."
                )
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask", None)
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Some model wrappers accept attention_mask; fall back if not.
        try:
            logits = forward_model(input_ids, attention_mask=attention_mask)
        except TypeError:
            logits = forward_model(input_ids)

        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        if logits.ndim != 3:
            raise ValueError(f"Expected logits [B, seq, vocab]; got {tuple(logits.shape)}")
        if int(logits.shape[-1]) != vocab:
            raise ValueError(f"Vocab mismatch: logits vocab={logits.shape[-1]} vs R_logit vocab={vocab}")

        B, T, _ = logits.shape

        # Choose token position
        if isinstance(position, str):
            pos_str = position.lower()
            if pos_str == "last":
                if attention_mask is not None:
                    idx = attention_mask.to(torch.long).sum(dim=1) - 1
                else:
                    if pad_token_id is not None:
                        lengths = (input_ids != pad_token_id).to(torch.long).sum(dim=1)
                        idx = lengths - 1
                    else:
                        idx = torch.full((B,), T - 1, device=device, dtype=torch.long)
                idx = torch.clamp(idx, min=0, max=T - 1)
            elif pos_str == "first":
                idx = torch.zeros((B,), device=device, dtype=torch.long)
            else:
                try:
                    pos_i = int(position)
                except Exception as e:
                    raise ValueError(f"Unsupported position='{position}'. Use 'last', 'first', or an integer.") from e
                if pos_i < 0:
                    pos_i = T + pos_i
                idx = torch.full((B,), pos_i, device=device, dtype=torch.long)
        else:
            pos_i = int(position)
            if pos_i < 0:
                pos_i = T + pos_i
            idx = torch.full((B,), pos_i, device=device, dtype=torch.long)

        ar = torch.arange(B, device=device)
        chosen_logits = logits[ar, idx, :].float()
        p = F.softmax(chosen_logits, dim=-1)  # [B, vocab], float32

        p_sum += p.sum(dim=0).to(dtype)

        A = p @ R_logit  # [B, m]
        A64 = A.to(dtype)
        B_sum += A64.T @ A64
        N += int(B)

        del logits, chosen_logits, p, A, A64

    if N == 0:
        raise RuntimeError("No batches were processed (N=0). Check dataloader and max_batches.")

    p_bar = p_sum / float(N)  # [vocab]
    p_sqrt = torch.sqrt(torch.clamp(p_bar, min=0.0)).to(dtype=R_logit.dtype)

    # G1 = R^T diag(p_bar) R = (R * sqrt(p_bar))^T (R * sqrt(p_bar))
    Rw = R_logit * p_sqrt.unsqueeze(1)  # [vocab, m]
    G1 = (Rw.T @ Rw).to(dtype)

    # G2 = E[a a^T]
    G2 = B_sum / float(N)

    G = G1 - G2
    G = 0.5 * (G + G.T)

    if return_N:
        return G, int(N)
    return G


@torch.no_grad()
def build_fisher_D(G: torch.Tensor, ridge: float = 1e-3) -> torch.Tensor:
    """
    ZCA whitening:
        eigvals, eigvecs = eigh(G + ridge*I)
        D = eigvecs @ diag(eigvals**(-0.5)) @ eigvecs^T
    """
    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError(f"G must be square [m,m]; got {tuple(G.shape)}")

    m = int(G.shape[0])
    I = torch.eye(m, device=G.device, dtype=G.dtype)
    G_reg = G + float(ridge) * I

    eigvals, eigvecs = torch.linalg.eigh(G_reg)
    eigvals = torch.clamp(eigvals, min=float(ridge))
    D = eigvecs @ torch.diag(eigvals.pow(-0.5)) @ eigvecs.T
    D = 0.5 * (D + D.T)
    return D


@torch.no_grad()
def apply_D(V_write: torch.Tensor, D: torch.Tensor, renorm_cols: bool = True) -> torch.Tensor:
    """
    Apply D in residual write-space:
        V_dec = V_write @ D
    """
    if V_write.ndim != 2:
        raise ValueError(f"V_write must be [d, m]; got {tuple(V_write.shape)}")
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"D must be square [m,m]; got {tuple(D.shape)}")
    if V_write.shape[1] != D.shape[0]:
        raise ValueError(f"Shape mismatch: V_write is {tuple(V_write.shape)}, D is {tuple(D.shape)}")

    # V_dec = V_write @ D
    # Ensure matmul operands have same dtype/device
    if D.device != V_write.device:
        D = D.to(device=V_write.device)

    # Promote V_write to D.dtype for matmul, then optionally cast back
    orig_dtype = V_write.dtype
    if V_write.dtype != D.dtype:
        V_write = V_write.to(dtype=D.dtype)

    V_dec = V_write @ D

    if renorm_cols:
        norms = torch.linalg.norm(V_dec, dim=0, keepdim=True).clamp(min=1e-12)
        V_dec = V_dec / norms

    # Cast back so downstream stays lightweight (R_logit stays float32 typically)
    if V_dec.dtype != orig_dtype:
        V_dec = V_dec.to(dtype=orig_dtype)

    if renorm_cols:
        norms = torch.linalg.norm(V_dec, dim=0, keepdim=True).clamp(min=1e-12)
        V_dec = V_dec / norms
    return V_dec


@torch.no_grad()
def interference_metrics(G: torch.Tensor) -> Dict[str, float]:
    """
    Report simple overlap/interference statistics for a (near-)symmetric Gram matrix.
    """
    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError(f"G must be square [m,m]; got {tuple(G.shape)}")

    m = int(G.shape[0])
    diag = torch.diag(G)
    off = G - torch.diag(diag)

    fro = torch.linalg.norm(G).item()
    off_fro = torch.linalg.norm(off).item()
    off_ratio = off_fro / max(fro, 1e-12)

    abs_off = off.abs()
    if m > 1:
        max_abs_off = abs_off.max().item()
        mean_abs_off = abs_off.sum().item() / float(m * m - m)
    else:
        max_abs_off = 0.0
        mean_abs_off = 0.0

    eigvals = torch.linalg.eigvalsh(0.5 * (G + G.T))
    eigvals = torch.clamp(eigvals, min=1e-12)
    cond = (eigvals.max() / eigvals.min()).item()

    return {
        "fro": float(fro),
        "offdiag_fro": float(off_fro),
        "offdiag_ratio": float(off_ratio),
        "max_abs_offdiag": float(max_abs_off),
        "mean_abs_offdiag": float(mean_abs_off),
        "cond_number": float(cond),
    }


# -------------------------
# __main__ self-test (synthetic)
# -------------------------

class _DummyTokenizer:
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id


class _DummyModel:
    """
    Minimal model that returns random logits, to exercise fisher_gram_streaming.
    """
    def __init__(self, vocab: int, seq_len: int = 4, seed: int = 0, device: str = "cpu"):
        self.vocab = vocab
        self.seq_len = seq_len
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.tokenizer = _DummyTokenizer(pad_token_id=0)

    def __call__(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = int(input_ids.shape[0])
        logits = torch.randn(B, self.seq_len, self.vocab, generator=self.generator, device=self.device)
        # Add mild vocab bias so p_bar is not uniform
        logits = logits + 0.2 * torch.linspace(-1, 1, steps=self.vocab, device=self.device)[None, None, :]
        return logits


def _self_test() -> None:
    device = torch.device("cpu")
    vocab = 64
    m = 16
    seq_len = 5
    batches = 25
    B = 32

    R = torch.randn(vocab, m, device=device, dtype=torch.float32)
    model = _DummyModel(vocab=vocab, seq_len=seq_len, seed=0, device="cpu")

    data = []
    for _ in range(batches):
        input_ids = torch.randint(0, 10, (B, seq_len), device=device)
        attention_mask = torch.ones_like(input_ids, device=device, dtype=torch.bool)
        data.append({"input_ids": input_ids, "attention_mask": attention_mask})

    G, _ = fisher_gram_streaming(R, model, data, device=device, position="last", max_batches=batches, return_N=True)
    D = build_fisher_D(G, ridge=1e-3)
    Gw = D.T @ G @ D

    mb = interference_metrics(G)
    ma = interference_metrics(Gw)

    print("[self-test] offdiag_ratio before:", mb["offdiag_ratio"])
    print("[self-test] offdiag_ratio after :", ma["offdiag_ratio"])
    if ma["offdiag_ratio"] > mb["offdiag_ratio"]:
        raise RuntimeError("Self-test failed: whitening did not reduce offdiag_ratio.")


if __name__ == "__main__":
    _self_test()
