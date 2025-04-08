import os

work_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(work_dir)

import yaml
import torch

from ...transformer import TransformerConfig, Transformer, generate_causal_mask
from .data import en_tokenizer, zh_tokenizer, collactor


class Inferencer:
    def __init__(self, config, ckpt, device=0, max_length=128) -> None:
        self.config = config
        self.device = torch.device(device)
        self.max_length = max_length

        self.load_model(ckpt)

    def load_model(self, ckpt):
        model_config = TransformerConfig(
            src_vocab_size=en_tokenizer.vocab_size,
            tgt_vocab_size=zh_tokenizer.vocab_size,
            **self.config["model"],
        )

        model = Transformer(model_config).eval()
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))

        self.model = model.to(self.device)

    def translate(self, en_text):
        self.model.eval()

        with torch.no_grad():
            en_tokens = collactor.encode_english([en_text])

            src = en_tokens["input_ids"].to(self.device)
            src_padding_mask = en_tokens["attention_mask"].to(self.device)

            zh_tokens = self.model.inference(
                src,
                tgt_start_token_id=zh_tokenizer.cls_token_id,
                tgt_end_token_id=zh_tokenizer.sep_token_id,
                src_key_padding_mask=src_padding_mask,
                src_attn_mask=None,
            )

            zh_text = zh_tokenizer.decode(
                zh_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )

        return zh_text


if __name__ == "__main__":
    with open("./config.yaml", "r") as f:
        config = yaml.safe_load(f)

    inferencer = Inferencer(
        config,
        ckpt="/data1/glcheng/download/translation-dataset/checkpoints/transformer.pth",
        device=6,
        max_length=128,
    )

    en_texts = [
        "new Questions Over California Water Project",
        "I had no idea it was a medal but my performance was the best I could have done , that is why I was so happy , that all the training and hard work had paid off .",
    ]

    for en_text in en_texts:
        zh_text = inferencer.translate(en_text)
        print(en_text, zh_text)
