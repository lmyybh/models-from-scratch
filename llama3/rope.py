from typing import Tuple
import torch
from torch import nn, Tensor


def apply_scaling(
    freqs: Tensor,
    scale_factor: float = 8.0,
    low_freq_factor: float = 1.0,
    high_freq_factor: float = 4.0,
    old_context_len: int = 8192,
) -> Tensor:
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * torch.pi / freqs
    new_freqs = torch.where(wavelen > low_freq_wavelen, freqs / scale_factor, freqs)
    smooth = (old_context_len / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    return torch.where(
        (wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen),
        (1 - smooth) * new_freqs / scale_factor + smooth * new_freqs,
        new_freqs,
    )


def precompute_freqs_cis(
    dim: int,
    seq_len: int,
    theta: float = 10000.0,
    use_scaled: bool = False,
    scale_factor: float = 8.0,
    low_freq_factor: float = 1.0,
    high_freq_factor: float = 4.0,
    old_context_len: int = 8192,
) -> Tensor:
    """预先计算 RoPE 中的 cosmθi 和 sinmθi

    Args:
        dim (int): query 或 key 对应的 head_dim
        seq_len (int): 输入序列长度
        theta (float, optional): θi 中底数的值. Defaults to 10000.0.
        use_scaled (bool, optional): 是否缩放频率. Defaults to False.
        scale_factor (float, optional): 频率缩放的倍数. Defaults to 8.0.
        low_freq_factor (float, optional): 低频因子. Defaults to 1.0.
        high_freq_factor (float, optional): 高频因子. Defaults to 4.0.
        old_context_len (int, optional): 频率缩放用到的最大文本长度. Defaults to 8192.

    Returns:
        Tensor: 使用极坐标 (1, mθi) 表示的 cosmθi 和 sinmθi
    """
    # 计算 θi
    freqs = 1.0 / theta ** (2 * torch.arange(0, dim // 2) / dim)
    # 频率缩放
    if use_scaled:
        freqs = apply_scaling(
            freqs, scale_factor, low_freq_factor, high_freq_factor, old_context_len
        )

    # 计算 m
    m = torch.arange(seq_len)
    # 计算 mθi
    freqs = torch.outer(m, freqs)
    # 极坐标表示 (1, mθi)，对应 cosmθi 和 sinmθi
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_cis


def apply_rotary_embedding_llama3(
    q: Tensor, k: Tensor, freqs_cis: Tensor
) -> Tuple[Tensor, Tensor]:
    """应用 RoPE 到 query 和 key

    Args:
        q (Tensor): [bz, h_q, seq_len, head_dim]
        k (Tensor): [bz, h_kv, seq_len, head_dim]
        freqs_cis (Tensor): [max_len, head_dim] 预先计算好的频率复数，每个复数表示 cosmθ + sinmθ i

    Returns:
        Tuple[Tensor, Tensor]: 使用 RoPE 后的 q 和 k
    """

    # 构造 (q0 + q1 i, q2 + q3 i, ...) 的复数形式, shape: [bz, h_q, seq_len, head_dim//2]
    q = torch.view_as_complex(q.reshape(*q.shape[:-1], -1, 2))
    # 构造 (k0 + k1 i, k2 + k3 i, ...) 的复数形式, shape: [bz, h_kv, seq_len, head_dim//2]
    k = torch.view_as_complex(k.reshape(*k.shape[:-1], -1, 2))

    # 复数 [bz, h, seq_len, head_dim//2] -> 两个实数 [bz, h, seq_len, head_dim//2, 2] -> 展开 [bz, h, seq_len, head_dim]
    q = torch.view_as_real(q * freqs_cis).flatten(-2)
    k = torch.view_as_real(k * freqs_cis).flatten(-2)

    return q, k


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_embedding_transformers(
    q: Tensor, k: Tensor, freqs_cis: Tensor
) -> Tuple[Tensor, Tensor]:
    """应用 RoPE 到 query 和 key

    Args:
        q (Tensor): [bz, h_q, seq_len, head_dim]
        k (Tensor): [bz, h_kv, seq_len, head_dim]
        freqs_cis (Tensor): [seq_len, head_dim] 预先计算好的频率复数，每个复数表示 cosmθ + sinmθ i

    Returns:
        Tuple[Tensor, Tensor]: 使用 RoPE 后的 q 和 k
    """
    seq_len, half_dim = freqs_cis.shape

    cos, sin = torch.split(torch.view_as_real(freqs_cis), 1, dim=-1)
    cos = cos.view(1, 1, seq_len, half_dim) # [1, 1, seq_len, head_dim / 2]
    sin = sin.view(1, 1, seq_len, half_dim) # [1, 1, seq_len, head_dim / 2]

    cos = (
        cos[:, :, :, None, :]
        .expand(1, 1, seq_len, 2, half_dim)
        .reshape(1, 1, seq_len, half_dim * 2)
    ) # [1, 1, seq_len, head_dim]
    sin = (
        sin[:, :, :, None, :]
        .expand(1, 1, seq_len, 2, half_dim)
        .reshape(1, 1, seq_len, half_dim * 2)
    ) # [1, 1, seq_len, head_dim]

    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)

    return q, k


def apply_rotary_embedding(
    q: Tensor, k: Tensor, freqs_cis: Tensor, method="llama3"
) -> Tuple[Tensor, Tensor]:
    assert method in {"llama3", "transformers"}

    if method == "llama3":
        return apply_rotary_embedding_llama3(q, k, freqs_cis)
    else:
        return apply_rotary_embedding_transformers(q, k, freqs_cis)
