from typing import Optional
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .configuration_transformer import TransformerConfig


class PositionalEncoding(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        # shape: [max_len, 1]
        pos = torch.arange(0, config.max_position_embeddings).unsqueeze(1)
        div_term = 1000 ** (
            -torch.arange(0, config.hidden_size, 2) / config.hidden_size
        )  # shape: [d / 2]

        pe = torch.zeros(config.max_position_embeddings, config.hidden_size)
        pe[:, 0::2] = torch.sin(pos * div_term)  # shape: [max_len, d / 2]
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, max_len, d]

        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, : x.size()[1]]


class TokenEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.lut = nn.Embedding(vocab_size, hidden_size)
        self.scaling = hidden_size**0.5

    def forward(self, text: Tensor) -> Tensor:
        return self.lut(text) * self.scaling


def generate_causal_mask(size):
    return torch.triu(torch.ones((size, size)), diagonal=1)


def attention_forward(
    module: nn.Module,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scaling: float,
    dropout: float = 0.0,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """attention

    Args:
        module (nn.Module):
        query (Tensor): [bz, h, len, d]
        key (Tensor): [bz, h, len, d]
        value (Tensor): [bz, h, len, d]
        scaling (float):
        dropout (float, optional): . Defaults to 0.0.
        mask (Optional[Tensor], optional): . Defaults to None.

    Returns:
        Tensor: [bz, h, len, d]
    """
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling
    if mask is not None:
        attn_weights.masked_fill_(mask, float("-inf"))

    attn_weights = torch.softmax(attn_weights, dim=-1)  # shape: [bz, h, len, len]
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value)  # shape: [bz, h, len, d]

    return attn_output


class MultiHeadAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.scaling = self.attention_head_size**-0.5
        self.attention_dropout = config.attention_dropout

        self.query = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.qkv_bias
        )
        self.key = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.qkv_bias
        )
        self.value = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.qkv_bias
        )

        self.linear = nn.Linear(self.all_head_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        # shape: [bz, len, h * d] -> [bz, len, h, d]
        new_size = x.size()[:2] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_size)

        # shape: [bz, len, h, d] -> [bz, h, len, d]
        return x.permute(0, 2, 1, 3)

    def merge_masks(
        self, key_padding_mask: Optional[Tensor], attn_mask: Optional[Tensor]
    ) -> Tensor | None:
        """merge two masks

        Args:
            key_padding_mask (Optional[Tensor]): padding mask for key, shape: [bz, L_k]
            attn_mask (Optional[Tensor]): attention mask, shape: [L_q, L_k]
        Returns:
            Tensor | None: _description_
        """
        if key_padding_mask is None and attn_mask is None:
            return None

        mask = None
        if key_padding_mask is not None:
            # shape: [bz, L_k] -> [bz, 1, 1, L_k] -> [bz, h, 1, L_k]
            bz, L_k = key_padding_mask.shape
            key_padding_mask = key_padding_mask.view(bz, 1, 1, L_k).expand(
                -1, self.num_attention_heads, -1, -1
            )
            mask = key_padding_mask

        if attn_mask is not None:
            # shape: [L_q, L_k] -> [1, 1, L_q, L_k] -> [bz, h, L_q, L_k]
            attn_mask = attn_mask.view(
                1, 1, attn_mask.size()[0], attn_mask.size()[1]
            ).expand(bz, self.num_attention_heads, -1, -1)

            if mask is None:
                mask = attn_mask
            else:
                mask = mask.logical_or(attn_mask)

        return mask

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        q = self.transpose_for_scores(self.query(q))
        k = self.transpose_for_scores(self.key(k))
        v = self.transpose_for_scores(self.value(v))

        mask = self.merge_masks(key_padding_mask, attn_mask)

        # shape: [bz, h, len, d]
        output = attention_forward(
            self,
            q,
            k,
            v,
            self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            mask=mask,
        )
        # shape: [bz, len, h, d]
        output = output.transpose(1, 2).contiguous()
        # shape: [bz, len, h * d]
        new_size = output.size()[:2] + (self.all_head_size,)
        output = output.reshape(new_size)

        output = self.linear(output)
        output = self.dropout(output)

        return output


class FeedForwardNetworks(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.dropout(x)

        return x


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x = (x - mean) / std + self.eps

        return x * self.gamma + self.beta


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.norm1 = LayerNorm(config.hidden_size)
        self.ffn = FeedForwardNetworks(config)
        self.norm2 = LayerNorm(config.hidden_size)

    def forward(
        self,
        src: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        src_attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        residual = src
        x = self.attention(
            q=src,
            k=src,
            v=src,
            key_padding_mask=src_key_padding_mask,
            attn_mask=src_attn_mask,
        )
        x += residual
        x = self.norm1(x)

        residual = x
        x = self.ffn(x)
        x += residual
        x = self.norm2(x)

        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.self_attention = MultiHeadAttention(config)
        self.norm1 = LayerNorm(config.hidden_size)

        self.cross_attention = MultiHeadAttention(config)
        self.norm2 = LayerNorm(config.hidden_size)

        self.ffn = FeedForwardNetworks(config)
        self.norm3 = LayerNorm(config.hidden_size)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_key_padding_mask: Optional[Tensor] = None,
        tgt_attn_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        memory_attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        residual = tgt
        x = self.self_attention(
            q=tgt,
            k=tgt,
            v=tgt,
            key_padding_mask=tgt_key_padding_mask,
            attn_mask=tgt_attn_mask,
        )
        x += residual
        x = self.norm1(x)

        residual = x
        x = self.cross_attention(
            q=x,
            k=memory,
            v=memory,
            key_padding_mask=memory_key_padding_mask,
            attn_mask=memory_attn_mask,
        )
        x += residual
        x = self.norm2(x)

        residual = x
        x = self.ffn(x)
        x += residual
        x = self.norm3(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(config) for _ in range(config.num_layers)]
        )

    def forward(
        self,
        src: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        src_attn_mask: Optional[Tensor] = None,
    ) -> tuple:
        x = src
        all_hidden_states = ()
        for i, layer in enumerate(self.layers):
            x = layer(x, src_key_padding_mask, src_attn_mask)

            all_hidden_states += (x,)

        return all_hidden_states


class TransformerDecoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(config) for _ in range(config.num_layers)]
        )

    def forward(
        self,
        tgt: Tensor,
        memory_tuple: tuple,
        tgt_key_padding_mask: Optional[Tensor] = None,
        tgt_attn_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        memory_attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = tgt
        for i, layer in enumerate(self.layers):
            x = layer(
                x,
                memory_tuple[i],
                tgt_key_padding_mask,
                tgt_attn_mask,
                memory_key_padding_mask,
                memory_attn_mask,
            )

        return x


class TransformerOutputLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.linear = nn.Linear(config.hidden_size, config.tgt_vocab_size)

    def forward(self, hidden_state: Tensor):
        return torch.softmax(self.linear(hidden_state), dim=-1)


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.input_embeddings = TokenEmbeddings(
            config.src_vocab_size, config.hidden_size
        )
        self.output_embeddings = TokenEmbeddings(
            config.tgt_vocab_size, config.hidden_size
        )
        self.positional_encoding = PositionalEncoding(config)
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)
        self.output = TransformerOutputLayer(config)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        src_attn_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        tgt_attn_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        memory_attn_mask: Optional[Tensor] = None,
    ):
        src = self.input_embeddings(src)
        src = self.positional_encoding(src)
        tgt = self.output_embeddings(tgt)
        tgt = self.positional_encoding(tgt)

        encoder_outputs = self.encoder(src, src_key_padding_mask, src_attn_mask)
        output = self.decoder(
            tgt,
            encoder_outputs,
            tgt_key_padding_mask,
            tgt_attn_mask,
            memory_key_padding_mask,
            memory_attn_mask,
        )
        output = self.output(output)

        return output


if __name__ == "__main__":
    from ..utils import generate_batch_text_tokens

    config = TransformerConfig()

    device = torch.device(0)
    pad_index = 0

    src = generate_batch_text_tokens(
        [4, 8, 6], max_len=10, vocab_size=config.src_vocab_size, pad_index=pad_index
    ).to(device)
    src_padding_mask = src == pad_index

    tgt = generate_batch_text_tokens(
        [6, 8, 12], max_len=12, vocab_size=config.tgt_vocab_size, pad_index=pad_index
    ).to(device)
    tgt_padding_mask = tgt == pad_index

    causal_mask = generate_causal_mask(tgt.size()[1]).to(device)

    model = Transformer(config).to(device)

    output = model.forward(
        src,
        tgt,
        src_key_padding_mask=src_padding_mask,
        src_attn_mask=None,
        tgt_key_padding_mask=tgt_padding_mask,
        tgt_attn_mask=causal_mask,
        memory_key_padding_mask=src_padding_mask,
        memory_attn_mask=None,
    )
    print(output.shape)
