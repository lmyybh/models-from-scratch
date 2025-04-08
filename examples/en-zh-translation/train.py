import os

work_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(work_dir)

import logging
import yaml
from tqdm import tqdm
import random
import numpy as np
import torch

from ...transformer import TransformerConfig, Transformer, generate_causal_mask
from .data import en_tokenizer, zh_tokenizer, collactor, train_dataloader

logger = logging.getLogger(__name__)


def init_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Trainer:
    def __init__(self, config) -> None:
        self.config = config
        self.parse_config()

        init_seed(self.seed)

        self.load_model()
        self.load_optimizer()
        self.load_criterion()

    def parse_config(self):
        self.seed = self.config["train"]["seed"]
        self.epochs = self.config["train"]["epochs"]
        self.device = torch.device(self.config["train"]["gpu"])
        self.lr = self.config["train"]["lr"]
        self.output_folder = self.config["train"]["output_folder"]
        self.max_length = self.config["tokenizer"]["max_length"]

    def load_model(self):
        model_config = TransformerConfig(
            src_vocab_size=en_tokenizer.vocab_size,
            tgt_vocab_size=zh_tokenizer.vocab_size,
            **self.config["model"],
        )

        model = Transformer(model_config).to(self.device)

        self.model = model

    def load_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-9
        )

    def load_criterion(self):
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

    def save_checkpoint(self):
        checkpoint_file = os.path.join(self.output_folder, "transformer.pth")
        torch.save(self.model.state_dict(), checkpoint_file)
        logger.info("Save:", checkpoint_file)

    def train(self, train_dataloader):
        for epoch in range(self.epochs):
            # train
            with tqdm(total=len(train_dataloader), ncols=150) as _tqdm:
                _tqdm.set_description(f"epoch: {epoch+1}/{self.epochs}")

                self.model.train()
                for batch_idx, data in enumerate(train_dataloader):
                    self.train_batch(batch_idx, data, _tqdm)

            # save checkpoint
            self.save_checkpoint()

            # evalution
            self.evaluation()

    def train_batch(self, batch_idx, data, _tqdm):
        src = data["en"]["input_ids"].to(self.device)
        src_padding_mask = data["en"]["attention_mask"].to(self.device)

        tgt = data["zh"]["input_ids"].to(self.device)
        tgt_padding_mask = data["zh"]["attention_mask"].to(self.device)

        causal_mask = generate_causal_mask(tgt.size()[1] - 1).to(self.device)

        output = self.model(
            src,
            tgt[:, :-1],  # shifted right
            src_key_padding_mask=src_padding_mask,
            src_attn_mask=None,
            tgt_key_padding_mask=tgt_padding_mask[:, :-1],
            tgt_attn_mask=causal_mask,
            memory_key_padding_mask=src_padding_mask,
            memory_attn_mask=None,
        )

        loss = self.criterion(
            output.view(-1, output.size()[-1]), tgt[:, 1:].reshape(-1)
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        _tqdm.set_postfix(loss=f"{loss.item():.4f}")
        _tqdm.update(1)

    def evaluation(self):
        self.model.eval()

        print()

        en_texts = [
            "new Questions Over California Water Project",
            "I had no idea it was a medal but my performance was the best I could have done , that is why I was so happy , that all the training and hard work had paid off .",
        ]
        for en_text in en_texts:
            zh_text = self.translate(en_text)
            print(en_text, zh_text)

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
    # load config
    with open("./config.yaml", "r") as f:
        config = yaml.safe_load(f)

    trainer = Trainer(config)
    trainer.train(train_dataloader)
