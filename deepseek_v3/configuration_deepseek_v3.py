from ..utils import ModelConfig


class DeepSeekV3Config(ModelConfig):

    model_name = "deepseek_v3"

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 512,
        intermediate_size: int = 1024,
        num_hidden_layers: int = 8,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 8,
        q_lora_rank: int = 256,
        kv_lora_rank: int = 128,
        qk_rope_head_dim: int = 64,
        qk_nope_head_dim: int = 128,
        v_head_dim: int = 128,
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        n_shared_experts: int = 1,
        n_routed_experts: int = 8,
        n_group: int = 8,
        topk_group: int = 4,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 2.5,
        num_experts_per_token: int = 8,
    ):
        """关于 DeepSeekV3 模型的配置信息

        Args:
            vocab_size (int): 词汇表大小
            hidden_size (int, optional): 隐藏层的维度. Defaults to 512.
            intermediate_size (int, optional): 中间层的维度. Defaults to 1024.
            num_hidden_layers (int, optional): 解码器的层数. Defaults to 8.
            num_attention_heads (int, optional): 注意力头的数量. Defaults to 8.
            num_key_value_heads (int, optional): key 和 value 注意力头的数量. Defaults to 8.
            q_lora_rank (int, optional): 查询投影的 LoRA 秩. Defaults to 256.
            kv_lora_rank (int, optional): key 和 value 投影的 LoRA 秩. Defaults to 128.
            qk_rope_head_dim (int, optional): 查询和键的 RoPE 头维度. Defaults to 64.
            qk_nope_head_dim (int, optional): 查询和键的 NOPE 头维度. Defaults to 128.
            v_head_dim (int, optional): 值头的维度. Defaults to 128.
            rms_norm_eps (float, optional): RMSNorm 的 epsilon 值. Defaults to 1e-6.
            attention_bias (bool, optional): 是否在注意力层中使用偏置. Defaults to False.
            attention_dropout (float, optional): 注意力层的 dropout 概率. Defaults to 0.0.
            n_shared_experts (int, optional): 共享专家的数量. Defaults to 1.
            n_routed_experts (int, optional): 路由专家的数量. Defaults to 8.
            n_group (int, optional): 路由组的数量. Defaults to 8.
            topk_group (int, optional): 每组选择的 top-k 数量. Defaults to 4.
            norm_topk_prob (bool, optional): 是否对 top-k 概率进行归一化. Defaults to True.
            routed_scaling_factor (float, optional): 路由缩放因子. Defaults to 2.5.
            num_experts_per_token (int, optional): 每个 token 分配的专家数量. Defaults to 8.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_head_dim = qk_rope_head_dim + qk_nope_head_dim
        self.v_head_dim = v_head_dim
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.num_experts_per_token = num_experts_per_token
