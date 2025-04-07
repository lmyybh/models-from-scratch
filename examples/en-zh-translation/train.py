import os

work_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(work_dir)

import yaml
from tqdm import tqdm
import random
import numpy as np
import torch

from ...transformer import TransformerConfig, Transformer, generate_causal_mask
from .data import en_tokenizer, zh_tokenizer, train_dataloader


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

    def load_model(self):
        model_config = TransformerConfig(
            src_vocab_size=en_tokenizer.vocab_size,
            tgt_vocab_size=zh_tokenizer.vocab_size,
            **self.config["model"]
        )

        model = Transformer(model_config).to(self.device)

        self.model = model

    def load_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-9
        )

    def load_criterion(self):
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

    def train(self, train_dataloder):
        self.model.train()
        for epoch in range(self.epochs):
            for batch_idx, data in enumerate(tqdm(train_dataloder, ncols=100)):
                self.train_batch(batch_idx, data)

            break

    def train_batch(self, batch_idx, data):
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
        print(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    # load config
    with open("./config.yaml", "r") as f:
        config = yaml.safe_load(f)

    trainer = Trainer(config)
    trainer.train(train_dataloader)
