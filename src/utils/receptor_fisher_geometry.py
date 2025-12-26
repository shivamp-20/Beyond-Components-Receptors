import torch
import torch.nn.functional as F


def make_logit_receptors(V_write: torch.Tensor, W_U_raw: torch.Tensor) -> torch.Tensor:
    """
    Project write directions through unembedding to get logit receptors.
    Args:
        V_write: [d_model, m] (or augmented dim)
        W_U_raw: [vocab, d_model] or [d_model, vocab]
    Returns:
        R_logit: [vocab, m]
    """
    if W_U_raw.ndim != 2:
        raise ValueError("W_U_raw must be 2D")
    if V_write.ndim != 2:
        raise ValueError("V_write must be 2D")
    if W_U_raw.shape[0] == V_write.shape[0]:
        # [d_model, vocab]
        W_U_paper = W_U_raw
    elif W_U_raw.shape[1] == V_write.shape[0]:
        # [vocab, d_model]
        W_U_paper = W_U_raw.T
    else:
        raise ValueError(f"W_U_raw shape {W_U_raw.shape} not compatible with V_write shape {V_write.shape}")
    R_rows = V_write.T @ W_U_paper  # [m, vocab]
    R_logit = R_rows.T  # [vocab, m]
    return R_logit.to(dtype=torch.float32 if R_logit.dtype.is_floating_point else torch.float64)


def fisher_gram_streaming(R_logit: torch.Tensor, model, dataloader, device,
                         position="last", max_batches=50,
                         return_N=True, dtype=torch.float64):
    """
    Compute Fisher/KL Gram matrix for logit receptors using streaming.
    Args:
        R_logit: [vocab, m]
        model: model with logits output
        dataloader: yields batches of input_ids
        device: torch.device
        position: 'last' or int (token position)
        max_batches: int
        return_N: bool
        dtype: torch dtype
    Returns:
        (G, N) or G
    """
    m = R_logit.shape[1]
    vocab = R_logit.shape[0]
    p_sum = torch.zeros(vocab, dtype=dtype, device=device)
    B_sum = torch.zeros(m, m, dtype=dtype, device=device)
    N = 0
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        input_ids = batch['input_ids'] if isinstance(batch, dict) else batch[0]
        input_ids = input_ids.to(device)
        with torch.no_grad():
            logits = model(input_ids)
            if isinstance(logits, tuple):
                logits = logits[0]
            # logits: [B, L, vocab] or [B, vocab]
            if logits.ndim == 3:
                if position == "last":
                    logits = logits[:, -1, :]
                else:
                    logits = logits[:, position, :]
            p = F.softmax(logits, dim=-1).to(dtype)
        p_sum += p.sum(dim=0)
        A = p @ R_logit  # [B, m]
        B_sum += A.T @ A
        N += p.shape[0]
    p_bar = p_sum / N
    sqrt_pbar = torch.sqrt(p_bar)
    G1 = (R_logit * sqrt_pbar[:, None]).T @ (R_logit * sqrt_pbar[:, None])
    G2 = B_sum / N
    G = G1 - G2
    G = 0.5 * (G + G.T)
    return (G, N) if return_N else G


def build_fisher_D(G: torch.Tensor, ridge: float = 1e-3) -> torch.Tensor:
    """
    ZCA whitening for Fisher geometry.
    Args:
        G: [m, m] Fisher Gram
        ridge: float
    Returns:
        D: [m, m] whitening matrix
    """
    m = G.shape[0]
    G_ridge = G + ridge * torch.eye(m, dtype=G.dtype, device=G.device)
    eigvals, eigvecs = torch.linalg.eigh(G_ridge)
    eigvals = torch.clamp(eigvals, min=ridge)
    D = eigvecs @ torch.diag(eigvals ** -0.5) @ eigvecs.T
    return D


def apply_D(V_write: torch.Tensor, D: torch.Tensor, renorm_cols=True) -> torch.Tensor:
    """
    Apply whitening D to write directions.
    Args:
        V_write: [d_model, m]
        D: [m, m]
        renorm_cols: bool
    Returns:
        V_dec: [d_model, m]
    """
    V_dec = V_write @ D
    if renorm_cols:
        V_dec = F.normalize(V_dec, dim=0)
    return V_dec


def interference_metrics(G: torch.Tensor) -> dict:
    """
    Compute interference metrics for Gram matrix.
    Args:
        G: [m, m]
    Returns:
        dict of metrics
    """
    m = G.shape[0]
    offdiag = G - torch.diag(torch.diagonal(G))
    fro = torch.norm(G, p='fro').item()
    offdiag_fro = torch.norm(offdiag, p='fro').item()
    max_abs_offdiag = offdiag.abs().max().item()
    mean_abs_offdiag = offdiag.abs().sum().item() / (m * (m - 1))
    eigvals = torch.linalg.eigvalsh(G)
    cond_number = (eigvals.max().abs() / (eigvals.min().abs() + 1e-8)).item()
    return dict(
        fro=fro,
        offdiag_fro=offdiag_fro,
        offdiag_frac=offdiag_fro / fro if fro > 0 else float('nan'),
        max_abs_offdiag=max_abs_offdiag,
        mean_abs_offdiag=mean_abs_offdiag,
        cond_number=cond_number,
    )


if __name__ == "__main__":
    torch.manual_seed(42)
    vocab = 50
    m = 8
    d_model = 32
    R_logit = torch.randn(vocab, m)
    # Simulate random simplex-like p
    B = 1000
    p_batches = F.softmax(torch.randn(B, vocab), dim=-1)
    # Gram by streaming
    p_sum = p_batches.sum(dim=0)
    A = p_batches @ R_logit
    B_sum = A.T @ A
    N = B
    p_bar = p_sum / N
    sqrt_pbar = torch.sqrt(p_bar)
    G1 = (R_logit * sqrt_pbar[:, None]).T @ (R_logit * sqrt_pbar[:, None])
    G2 = B_sum / N
    G = G1 - G2
    G = 0.5 * (G + G.T)
    D = build_fisher_D(G)
    G_dec = D.T @ G @ D
    print("Original interference:", interference_metrics(G))
    print("Whitened interference:", interference_metrics(G_dec))
