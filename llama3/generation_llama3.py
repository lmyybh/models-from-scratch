from typing import Optional
import torch
import safetensors.torch
from collections import OrderedDict
from modelscope import AutoTokenizer

from .configuration_llama3 import Llama3Config
from .modeling_llama3 import Llama3


class LlamaModel:
    def __init__(self, config: Llama3Config, tokenizer: AutoTokenizer, model: Llama3, device: str = "cuda"):
        """LlamaModel 的初始化方法

        Args:
            tokenizer (AutoTokenizer): 推理用到的 tokenizer
            model (Llama3): 推理用到的 llama 模型
            device (str, optional): 推理的设备. Defaults to "cuda".
        """
        self.config = config
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    @staticmethod
    def build_1B(ckpt_file: str, device: str = "cuda") -> "LlamaModel":
        """搭建 llama3_1B 模型

        Args:
            ckpt_file (str): llama3_1B 的 safetensors 模型文件路径，对应 modelscope 中的 "LLM-Research/Llama-3.2-1B"
            device (str, optional): 推理的设备. Defaults to "cuda".

        Returns:
            LlamaModel: llama3_1B 模型
        """
        config = Llama3Config(
            vocab_size=128256,
            num_layers=16,
            hidden_size=2048,
            num_attention_heads=32,
            num_kv_heads=8,
            intermediate_size=8192,
            max_seq_len=512,
            rope_method="transformers",
            rope_theta=500000.0,
            rope_scaling=True,
            rope_scale_factor=32.0,
            rope_low_freq_factor=1.0,
            rope_high_freq_factor=4.0,
            rope_old_context_len=8192,
        )
        model = Llama3(config)
        llama3_dict = LlamaModel._convert_weigths_1B(ckpt_file)
        model.load_state_dict(llama3_dict)
        model = model.to(device).eval()

        tokenizer = AutoTokenizer.from_pretrained("LLM-Research/Llama-3.2-1B")

        return LlamaModel(config, tokenizer, model, device)

    @staticmethod
    def _convert_weigths_1B(ckpt_file: str) -> OrderedDict:
        """转换 "LLM-Research/Llama-3.2-1B" 的 safetensors 模型文件

        Args:
            ckpt_file (str): llama3_1B 的 safetensors 模型文件路径，对应 modelscope 中的 "LLM-Research/Llama-3.2-1B"

        Returns:
            OrderedDict: llama3_1B 的参数字典
        """
        key_map = {
            "model.embed_tokens.weight": "token_embedding.lut.weight",
            "model.norm.weight": "output.norm.gamma",
            "layers": {
                "input_layernorm.weight": "attention_norm.gamma",
                "mlp.down_proj.weight": "ffn.w2.weight",
                "mlp.gate_proj.weight": "ffn.w3.weight",
                "mlp.up_proj.weight": "ffn.w1.weight",
                "post_attention_layernorm.weight": "ffn_norm.gamma",
                "self_attn.k_proj.weight": "attention.wk.weight",
                "self_attn.o_proj.weight": "attention.wo.weight",
                "self_attn.q_proj.weight": "attention.wq.weight",
                "self_attn.v_proj.weight": "attention.wv.weight",
            },
        }

        state_dict = safetensors.torch.load_file(ckpt_file)
        llama3_dict = OrderedDict()

        for k, v in state_dict.items():
            if k in key_map:
                new_key = key_map[k]
            else:
                assert "model.layers" in k, f"unknow key: {k}"

                splits = k.split(".")
                prefix, suffix = ".".join(splits[:3]), ".".join(splits[3:])

                new_key = prefix.replace("model.", "") + "." + key_map["layers"][suffix]

            llama3_dict[new_key] = v.to(torch.float32)

        # 输入和输出共享参数
        llama3_dict["output.linear.weight"] = llama3_dict["token_embedding.lut.weight"]

        return llama3_dict

    @torch.inference_mode()
    def generate(self, prompt_text: str, max_length: Optional[int] = None) -> str:
        """输入 prompt, 生成后续文本

        Args:
            prompt_text (str): 输入的 prompt
            max_length (Optional[int], optional): 生成文本的最大长度. Defaults to None.

        Returns:
            str: 生成的文本
        """
        tokens = self.tokenizer.batch_encode_plus([prompt_text], return_tensors="pt")
        tokens = tokens["input_ids"].to(self.device)
        output = self.model.generate(
            tokens, stop_token_id=self.tokenizer.eos_token_id, max_length=max_length
        )[0]
        output = self.tokenizer.decode(
            output, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        return output
