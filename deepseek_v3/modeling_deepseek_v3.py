import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .configuration_deepseek_v3 import DeepSeekV3Config


class DeepSeekV3RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: Tensor) -> Tensor:
        return (
            hidden_states
            * torch.rsqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.eps)
            * self.weight
        )


def repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    batch_size, num_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    # [bz, n, seq_len, d] -> [bz, n, 1, seq_len, d] -> [bz, n, n_rep, seq_len, d]
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch_size, num_heads, n_rep, seq_len, head_dim
    )
    # [bz, n * n_rep, seq_len, d]
    return hidden_states.reshape(batch_size, num_heads * n_rep, seq_len, head_dim)


class DeepSeekV3Attention(nn.Module):
    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.config = config

        self.num_heads = config.num_attention_heads
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.attention_dropout = config.attention_dropout
        self.attention_bias = config.attention_bias

        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_head_dim = config.qk_head_dim
        self.v_head_dim = config.v_head_dim

        self.q_down_proj = nn.Linear(
            config.hidden_size, config.q_lora_rank, bias=self.attention_bias
        )
        self.q_down_layernorm = DeepSeekV3RMSNorm(config.q_lora_rank)
        self.q_up_proj = nn.Linear(
            config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False
        )

        self.kv_down_proj_with_mqa = nn.Linear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=self.attention_bias,
        )
        self.kv_down_layernorm = DeepSeekV3RMSNorm(self.kv_lora_rank)
        self.kv_up_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            config.hidden_size,
            bias=self.attention_bias,
        )

        self.scaling = self.qk_head_dim ** (-0.5)
        # TODO: yarn

    def forward(self, hidden_states: Tensor):
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
        key_shape = (
            batch_size,
            seq_length,
            -1,
            self.qk_nope_head_dim + self.v_head_dim,
        )

        # [bz, seq_len, n * qk_dim]
        q_states = self.q_up_proj(
            self.q_down_layernorm(self.q_down_proj(hidden_states))
        )
        # [bz, seq_len, n * qk_dim] -> [bz, seq_len, n, qk_dim] -> [bz, n, seq_len, qk_dim]
        q_states = q_states.view(query_shape).transpose(1, 2)

        # q_pass: [bz, n, seq_len, qk_nope]
        # q_rot: [bz, n, seq_len, qk_rope]
        q_pass, q_rot = torch.split(
            q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # [bz, seq_len, kv_lora_rank + qk_rope]
        compressed_kv = self.kv_down_proj_with_mqa(hidden_states)

        # k_pass: [bz, seq_len, kv_lora_rank]
        # k_rot: [bz, seq_len, qk_rope]
        k_pass, k_rot = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        # [bz, seq_len, n * (qk_nope + v_dim)]
        k_pass = self.kv_up_proj(self.kv_down_layernorm(k_pass))

        # [bz, seq_len, n * (qk_nope + v_dim)] -> [bz, seq_len, n, qk_nope + v_dim] -> [bz, n, seq_len, qk_nope + v_dim]
        k_pass = k_pass.view(key_shape).transpose(1, 2)

        # k_pass: [bz, n, seq_len, qk_nope]
        # v_states: [bz, n, seq_len, v_dim]
        k_pass, v_states = torch.split(
            k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )

        # [bz, 1, seq_len, qk_rope]
        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        # TODO: Apply RoPE to q_rot and k_rot

        # [bz, 1, seq_len, qk_rope] -> [bz, n, seq_len, qk_rope]
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        # [bz, n, seq_len, qk_dim]
        q_states = torch.cat([q_pass, q_rot], dim=-1)

        # [bz, n, seq_len, qk_nope + qk_rope]
        k_states = torch.cat([k_pass, k_rot], dim=-1)

        # attention
        k_states = repeat_kv(k_states, self.num_key_value_groups)
        v_states = repeat_kv(v_states, self.num_key_value_groups)

        # [bz, n, seq_len, seq_len]
        attn_weights = torch.matmul(q_states, k_states.transpose(2, 3)) * self.scaling

        # TODO: Apply attention mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(
            attn_weights,
            p=0.0 if not self.training else self.attention_dropout,
            training=self.training,
        )

        # [bz, n, seq_len, v_dim]
        attn_output = torch.matmul(attn_weights, v_states)

        # [bz, n, seq_len, v_dim] -> [bz, seq_len, n, v_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()

        # [bz, seq_len, n, v_dim] -> [bz, seq_len, n * v_dim]
        attn_output = attn_output.reshape(batch_size, seq_length, -1).continuous()

        # [bz, seq_len, hidden_size]
        attn_output = self.o_proj(attn_output)

        return attn_output


class DeepSeekV3SharedMoE(nn.Module):
    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size * config.n_shared_experts
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = F.silu

    def forward(self, hidden_states: Tensor):
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class DeepSeekV3TopkRouter(nn.Module):
    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.config = config
        self.n_routed_experts = config.n_routed_experts

        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, config.hidden_size))
        )
        self.register_buffer(
            "e_score_correction_bias", torch.zeros(self.n_routed_experts)
        )

    def forward(self, hidden_states: Tensor):
        # [bz, seq_len, hidden_size] -> [bz * seq_len, hidden_size]
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        # [bz * seq_len, n_routed_experts]
        router_logits = F.linear(hidden_states, self.weight)
        return router_logits


class DeepSeekV3NaiveMoE(nn.Module):
    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_size, self.hidden_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, self.intermediate_size)
        )
        self.act_fn = F.silu

    def forward(
        self, hidden_states: Tensor, top_k_index: Tensor, top_k_weights: Tensor
    ):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue

            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(
                2, dim=-1
            )
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(
                current_hidden_states, self.down_proj[expert_idx]
            )
            current_hidden_states = (
                current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            )
            final_hidden_states.index_add_(0, token_idx, current_hidden_states)

        return final_hidden_states


class DeepSeekV3MoE(nn.Module):
    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.experts = DeepSeekV3NaiveMoE(config)
        self.gate = DeepSeekV3TopkRouter(config)
        self.shared_experts = DeepSeekV3SharedMoE(config)

        self.n_routed_experts = config.n_routed_experts
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.top_k = config.num_experts_per_token

    def route_tokens_to_experts(self, router_logits: Tensor):
        router_logits = router_logits.sigmoid()
        router_logits_for_choice = router_logits + self.gate.e_score_correction_bias
        group_scores = (
            router_logits_for_choice.view(
                -1, self.n_group, self.n_routed_experts // self.n_group
            )
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = router_logits_for_choice.masked_fill(
            ~score_mask.bool(), 0.0
        )
        topk_indices = torch.topk(
            scores_for_choice, k=self.top_k, dim=-1, sorted=False
        )[1]
        topk_weights = router_logits.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights

    def forward(self, hidden_states: Tensor):
        residual = hidden_states
        orig_shape = hidden_states.shape
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.experts(hidden_states, topk_indices, topk_weights).view(
            *orig_shape
        )
        hidden_states = hidden_states + self.shared_experts(residual)

        return hidden_states


class DeepSeekV3DecoderLayer(nn.Module):
    def __init__(self, config: DeepSeekV3Config):
        super().__init__()

        self.hidden_size = config.hidden_size

        self.self_attn = DeepSeekV3Attention(config)

        # TODO: MoE
        # self.mlp = DeepSeekV3MoE(config)

        self.input_layernorm = DeepSeekV3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = DeepSeekV3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(self, hidden_states: Tensor):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states
