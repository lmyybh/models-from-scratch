from typing import Optional
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .rope import precompute_freqs_cis, apply_rotary_embedding
from .configuration_llama3 import Llama3Config


def generate_casual_mask(size: int) -> Tensor:
    """生成 casusal mask

    Args:
        size (int): attention 矩阵维度

    Returns:
        Tensor: 下三角矩阵，0 表示显示，-inf 表示遮挡
    """
    return torch.triu(torch.full((size, size), float("-inf")), diagonal=1)


class RMSNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        """RMS Norm

        Args:
            normalized_shape (int): 归一化的维度
            eps (float, optional): 避免分母为 0 的附加值. Defaults to 1e-6.
        """
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: Tensor) -> Tensor:
        """前向计算

        Args:
            x (Tensor): 输入 Tensor

        Returns:
            Tensor: 归一化后的 Tensor
        """
        return (
            x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.gamma
        )


class TokenEmbeddings(nn.Module):
    def __init__(self, config: Llama3Config) -> None:
        """token 编码层

        Args:
            config (Llama3Config): 关于 llama3 模型的配置信息
        """
        super().__init__()

        self.lut = nn.Embedding(config.vocab_size, config.hidden_size)

    def forward(self, tokens: Tensor) -> Tensor:
        """前向计算

        Args:
            tokens (Tensor): 序列 tokens

        Returns:
            Tensor: 序列编码
        """
        return self.lut(tokens)


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """复制 n_rep 次 key 或 value

    Args:
        x (Tensor): shape: [bz, h_kv, seq_len, d]
        n_rep (int): 需要复制的次数

    Returns:
        Tensor: shape: [bz, h_kv * n_rep, seq_len, d]
    """
    bz, h_kv, seq_len, d = x.shape

    return (
        x[:, :, None, :, :]
        .expand(bz, h_kv, n_rep, seq_len, d)
        .reshape(bz, h_kv * n_rep, seq_len, d)
    )


class GroupQueryAttention(nn.Module):
    def __init__(self, config: Llama3Config) -> None:
        """Group Query Attention

        Args:
            config (Llama3Config): 关于 llama3 模型的配置信息
        """
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.n_rep = self.num_attention_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        self.scaling = config.head_dim**-0.5

        # RoPE
        self.rope_method = config.rope_method
        self.freqs_cis = precompute_freqs_cis(
            dim=self.head_dim,
            seq_len=config.max_seq_len,
            theta=config.rope_theta,
            use_scaled=config.rope_scaling,
            scale_factor=config.rope_scale_factor,
            low_freq_factor=config.rope_low_freq_factor,
            high_freq_factor=config.rope_high_freq_factor,
            old_context_len=config.rope_old_context_len,
        )

        self.wq = nn.Linear(
            self.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=config.qkv_bias,
        )
        self.wk = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.qkv_bias,
        )
        self.wv = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.qkv_bias,
        )

        self.wo = nn.Linear(
            self.num_attention_heads * self.head_dim,
            self.hidden_size,
            bias=config.mlp_bias,
        )

        self.cache_k = torch.zeros(
            config.max_batch_size, self.num_kv_heads, config.max_seq_len, self.head_dim
        )
        self.cache_v = torch.zeros(
            config.max_batch_size, self.num_kv_heads, config.max_seq_len, self.head_dim
        )

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        """调整多头 q, k, v 的维度，用于 attention 计算

        Args:
            x (Tensor): 输入的 query 或 key 或 value, shape: [bz, seq_len, h * d]

        Returns:
            Tensor: 调整维度后的 Tensor, shape: [bz, h, seq_len, d]
        """

        # [bz, seq_len, h*d] -> [bz, seq_len, h, d] -> [bz, h, seq_len, d]
        new_size = x.size()[:2] + (-1, self.head_dim)
        return x.view(new_size).transpose(1, 2)

    def forward(
        self, x: Tensor, start_pos: int, mask: Optional[Tensor] = None
    ) -> Tensor:
        """前向计算

        Args:
            x (Tensor): 输入的 x, shape: [bz, seq_len, hidden_size]
            start_pos (int): 输入序列的起始位置坐标
            mask (Optional[Tensor], optional): 下三角 causal mask, shape: [seq_len, cache_len+seq_len]. Defaults to None.

        Returns:
            Tensor: GQA 输出 Tensor, shape: [bz, seq_len, hidden_size]
        """

        bz, seq_len = x.shape[:2]

        q = self.transpose_for_scores(self.wq(x))  # [bz, h_q, seq_len, head_dim]
        k = self.transpose_for_scores(self.wk(x))  # [bz, h_kv, seq_len, head_dim]
        v = self.transpose_for_scores(self.wv(x))  # [bz, h_kv, seq_len, head_dim]

        self.freqs_cis = self.freqs_cis.to(q.device)
        q, k = apply_rotary_embedding(
            q,
            k,
            self.freqs_cis[start_pos : start_pos + seq_len],
            method=self.rope_method,
        )

        # kv cache
        self.cache_k = self.cache_k.to(k)
        self.cache_v = self.cache_v.to(v)

        self.cache_k[:bz, :, start_pos : start_pos + seq_len] = k
        self.cache_v[:bz, :, start_pos : start_pos + seq_len] = v

        # [bz, h_kv, cache_len+seq_len, head_dim]
        k = self.cache_k[:bz, :, : start_pos + seq_len]
        # [bz, h_kv, cache_len+seq_len, head_dim]
        v = self.cache_v[:bz, :, : start_pos + seq_len]

        k = repeat_kv(k, self.n_rep)  # [bz, h_q, cache_len+seq_len, head_dim]
        v = repeat_kv(v, self.n_rep)  # [bz, h_q, cache_len+seq_len, head_dim]

        # shape: [bz, h_q, seq_len, cache_len+seq_len]
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scaling

        if mask is not None:
            attn_weights += mask

        attn_weights = torch.softmax(attn_weights, dim=-1)

        # shape: [bz, h_q, seq_len, head_dim]
        output = torch.matmul(attn_weights, v)

        # shape: [bz, seq_len, h_q * head_dim]
        output = output.transpose(1, 2).contiguous().view(bz, seq_len, -1)
        # shape: [bz, seq_len, hidden_size]
        output = self.wo(output)

        return output


class FeedForwardNetworks(nn.Module):
    def __init__(self, config: Llama3Config) -> None:
        """前向网络

        Args:
            config (Llama3Config): 关于 llama3 模型的配置信息
        """

        super().__init__()

        self.w1 = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        self.w2 = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.mlp_bias
        )
        self.w3 = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )

    def forward(self, x: Tensor) -> Tensor:
        """前向计算

        Args:
            x (Tensor): 输入 Tensor, shape: [bz, seq_len, hidden_size]

        Returns:
            Tensor: 输出 Tensor, shape: [bz, seq_len, hidden_size]
        """
        return self.w2(F.silu(self.w3(x)) * self.w1(x))


class DecoderLayer(nn.Module):
    def __init__(self, config: Llama3Config) -> None:
        """llama3 解码层

        Args:
            config (Llama3Config): 关于 llama3 模型的配置信息
        """
        super().__init__()

        self.attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = GroupQueryAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn = FeedForwardNetworks(config)

    def forward(
        self, x: Tensor, start_pos: int, mask: Optional[Tensor] = None
    ) -> Tensor:
        """前向计算

        Args:
            x (Tensor): 输入的 x, shape: [bz, seq_len, hidden_size]
            start_pos (int): 输入序列的起始位置坐标
            mask (Optional[Tensor], optional): 下三角 causal mask, shape: [seq_len, cache_len + seq_len]. Defaults to None.

        Returns:
            Tensor: 输出 Tensor, shape: [bz, seq_len, hidden_size]
        """

        x = x + self.attention(self.attention_norm(x), start_pos, mask=mask)
        x = x + self.ffn(self.ffn_norm(x))

        return x


class Llama3Outputlayer(nn.Module):
    def __init__(self, config: Llama3Config) -> None:
        """llama3 输出层

        Args:
            config (Llama3Config): 关于 llama3 模型的配置信息
        """

        super().__init__()

        self.norm = RMSNorm(
            normalized_shape=config.hidden_size, eps=config.rms_norm_eps
        )
        self.linear = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """前向计算

        Args:
            x (Tensor): 输入 Tensor, shape: [bz, seq_len, hidden_size]

        Returns:
            Tensor: 输出 Tensor, shape: [bz, seq_len, vocab_size]
        """
        return self.linear(self.norm(x))


class Llama3(nn.Module):
    def __init__(self, config: Llama3Config) -> None:
        """llama3

        Args:
            config (Llama3Config): 关于 llama3 模型的配置信息
        """

        super().__init__()

        self.max_seq_len = config.max_seq_len
        self.initializer_range = config.initializer_range
        self.token_embedding = TokenEmbeddings(config)
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_layers)]
        )
        self.output = Llama3Outputlayer(config)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=self.initializer_range)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                m.weight.data.normal_(mean=0.0, std=self.initializer_range)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

    def forward(self, x: Tensor, start_pos: int) -> Tensor:
        """前向计算

        Args:
            x (Tensor): 输入的 x, shape: [bz, seq_len]
            start_pos (int): 输入序列的起始位置坐标

        Returns:
            Tensor: 输出 Tensor, shape: [bz, seq_len, hidden_size]
        """
        seq_len = x.shape[1]
        x = self.token_embedding(x)

        mask = None
        if seq_len > 1:
            # mask 是 下三角矩阵， 尺寸为 [seq_len, seq_len]
            mask = torch.full((seq_len, seq_len), float("-inf"), device=x.device)
            mask = torch.triu(mask, diagonal=1)

            # 当使用 kv-cache 时，需要拼接为 [seq_len, cache_len + seq_len]
            mask = torch.hstack(
                (torch.zeros((seq_len, start_pos), device=x.device), mask)
            ).type_as(x)

        for layer in self.layers:
            x = layer(x, start_pos, mask)

        return self.output(x)

    @torch.inference_mode()
    def generate(
        self, x: Tensor, stop_token_id: int, max_length: Optional[int] = None
    ) -> list:
        """推理生成

        Args:
            x (Tensor): 输入的 x, shape: [bz, seq_len]
            stop_token_id (int): 序列的结束 token id
            max_length (Optional[int], optional): 生成的最大长度. Defaults to None.

        Returns:
            list: 生成的 token_ids 列表
        """
        max_length = (
            self.max_seq_len
            if max_length is None
            else min(self.max_seq_len, max_length)
        )

        bz, seq_len = x.shape

        assert bz == 1, "batch_size must be 1"

        output_tokens = x[0].tolist()

        # prefill
        output = self.forward(x, start_pos=0)[0]  # [seq_len, vocab_size]
        next_token = output[-1].argmax()
        output_tokens.append(next_token.item())

        # decode
        for start_pos in range(seq_len, max_length - 1):
            output = self.forward(next_token.view(1, 1), start_pos=start_pos)[0]
            next_token = output[-1].argmax()
            output_tokens.append(next_token.item())

            if next_token.item() == stop_token_id:
                break

        return [output_tokens]


if __name__ == "__main__":
    from ..utils import generate_batch_text_tokens

    config = Llama3Config(vocab_size=128, max_seq_len=20)

    device = torch.device(0)
    model = Llama3(config).to(device)

    # forward
    x = generate_batch_text_tokens(
        [4, 8, 5], max_len=10, vocab_size=config.vocab_size, pad_index=0
    ).to(device)

    output = model(x, start_pos=0)
    print(output.shape)

    # generate
    model.eval()
    prompt = generate_batch_text_tokens(
        [5], max_len=5, vocab_size=config.vocab_size, pad_index=0
    ).to(device)
    tokens = model.generate(prompt, stop_token_id=100)
    print(prompt, tokens)
