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

            encoder_outputs = self.model.encode(src, src_padding_mask)

            zh_text = ""
            for _ in range(self.max_length):
                zh_tokens = collactor.encode_chinese([zh_text])

                tgt = zh_tokens["input_ids"][:, :-1].to(self.device)
                tgt_padding_mask = zh_tokens["attention_mask"][:, :-1].to(self.device)
                causal_mask = generate_causal_mask(tgt.size()[1]).to(self.device)

                output = self.model.decode(
                    tgt,
                    encoder_outputs,
                    tgt_key_padding_mask=tgt_padding_mask,
                    tgt_attn_mask=causal_mask,
                    memory_key_padding_mask=src_padding_mask,
                    memory_attn_mask=None,
                )
                token_id = output[0][-1].argmax().item()

                if token_id == zh_tokenizer.sep_token_id:
                    break

                word = zh_tokenizer.convert_ids_to_tokens(token_id)
                zh_text += word

        return zh_text


if __name__ == "__main__":    
    with open("./config.yaml", "r") as f:
        config = yaml.safe_load(f)

    inferencer = Inferencer(
        config,
        ckpt="/mnt/z/models/transformer/transformer.pth",
        device=0,
        max_length=128,
    )
    
    en_texts = [
        "new Questions Over California Water Project",
        "he was the brother that went with the flow .",
        "You actually have to implement the solution – and be willing to change course if it turns out that you did not know quite as much as you thought.",
    ]
    
    for en_text in en_texts:
        zh_text = inferencer.translate(en_text)
        print(en_text, zh_text)
