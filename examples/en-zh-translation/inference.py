import argparse
import torch

from ...transformer import TransformerConfig, Transformer
from .data import build_data
from ..utils import read_yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train model translated from English to Chinese"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Location of the config file"
    )
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Location of the checkpoint"
    )
    parser.add_argument("--gpu", type=int, default=0, help="Device Id")
    args = parser.parse_args()

    return args


class Inferencer:
    def __init__(self, config, ckpt, data_dict, device=0) -> None:
        self.config = config
        self.device = torch.device(device)
        self.parse_config()

        self.data_dict = data_dict
        self.load_model(ckpt)

    def parse_config(self):
        self.max_length = self.config["tokenizer"]["max_length"]

    def load_model(self, ckpt):
        model_config = TransformerConfig(
            src_vocab_size=self.data_dict["en_tokenizer"].vocab_size,
            tgt_vocab_size=self.data_dict["zh_tokenizer"].vocab_size,
            **self.config["model"],
        )
        print("Load model: \n" + str(model_config))

        model = Transformer(model_config).eval()
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        print("Load checkpoint: " + ckpt)

        self.model = model.to(self.device)

    def translate(self, en_text):
        self.model.eval()

        with torch.no_grad():
            en_tokens = self.data_dict["collactor"].encode_english([en_text])

            src = en_tokens["input_ids"].to(self.device)
            src_padding_mask = en_tokens["attention_mask"].to(self.device)

            zh_tokens = self.model.inference(
                src,
                tgt_start_token_id=self.data_dict["zh_tokenizer"].cls_token_id,
                tgt_end_token_id=self.data_dict["zh_tokenizer"].sep_token_id,
                src_key_padding_mask=src_padding_mask,
                src_attn_mask=None,
            )

            zh_text = self.data_dict["zh_tokenizer"].decode(
                zh_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )

        return zh_text


if __name__ == "__main__":
    args = parse_args()

    data_dict = build_data(args.config)

    config = read_yaml(args.config)

    inferencer = Inferencer(
        config,
        ckpt=args.ckpt,
        data_dict=data_dict,
        device=args.gpu,
    )

    en_texts = [
        "new Questions Over California Water Project",
        "I had no idea it was a medal but my performance was the best I could have done , that is why I was so happy , that all the training and hard work had paid off .",
    ]

    for en_text in en_texts:
        zh_text = inferencer.translate(en_text)
        print("-" * 100)
        print(f"[English]: {en_text}")
        print(f"[Chinese]: {zh_text}")
