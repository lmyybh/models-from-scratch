import os
import argparse
from tqdm import tqdm
import random
import numpy as np
import torch

from ..utils import init_seed, read_yaml, init_logger, dict2str
from ...transformer import TransformerConfig, Transformer, generate_causal_mask
from .data import build_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train model translated from English to Chinese"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Location of the config file"
    )
    args = parser.parse_args()

    return args


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
    def __init__(self, config, data_dict) -> None:
        self.config = config
        self.parse_config()

        init_seed(self.seed)

        self.build_logger()
        self.logger.info("Train config:\n" + dict2str(self.config))

        self.data_dict = data_dict
        self.load_model()
        self.load_optimizer()
        self.load_scheduler()
        self.load_criterion()

    def parse_config(self):
        self.seed = self.config["train"]["seed"]
        self.epochs = self.config["train"]["epochs"]
        self.device = torch.device(self.config["train"]["gpu"])
        self.lr = self.config["train"]["lr"]
        self.warmup_epochs = self.config["train"]["warmup_epochs"]
        self.lr_decay = self.config["train"]["lr_decay"]
        self.max_length = self.config["tokenizer"]["max_length"]
        self.pretrained_ckpt = self.config["train"].get("pretrained", None)
        self.output_folder = self.config["train"]["output_folder"]
        self.log_file = os.path.join(self.output_folder, "train.log")
        self.ckpt_folder = os.path.join(self.output_folder, "checkpoints")
        os.makedirs(self.ckpt_folder, exist_ok=True)

    def build_logger(self):
        self.logger = init_logger(self.log_file)

    def load_model(self):
        model_config = TransformerConfig(
            src_vocab_size=self.data_dict["en_tokenizer"].vocab_size,
            tgt_vocab_size=self.data_dict["zh_tokenizer"].vocab_size,
            **self.config["model"],
        )
        self.logger.info("Load model: \n" + str(model_config))

        model = Transformer(model_config)
        if self.pretrained_ckpt is not None:
            model.load_state_dict(torch.load(self.pretrained_ckpt, map_location="cpu"))
            self.logger.info("Load pretrained checkpoint: " + self.pretrained_ckpt)

        self.model = model.to(self.device)

    def load_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-9
        )

    def load_criterion(self):
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

    def load_scheduler(self):
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: (
                (epoch + 1) / self.warmup_epochs
                if epoch < self.warmup_epochs
                else self.lr_decay ** (epoch - self.warmup_epochs)
            ),
        )

    def save_checkpoint(self, epoch):
        checkpoint_file = os.path.join(self.ckpt_folder, f"model_{epoch+1}.pth")
        torch.save(self.model.state_dict(), checkpoint_file)
        self.logger.info("Save checkpoint: " + checkpoint_file)

    def train(self):
        train_dataloader = self.data_dict["train_dataloader"]
        for epoch in range(self.epochs):
            # train
            self.logger.info(f"epoch: {epoch+1}/{self.epochs}, start training...")

            self.model.train()
            with tqdm(total=len(train_dataloader), ncols=150) as _tqdm:
                _tqdm.set_description(f"epoch: {epoch+1}/{self.epochs}")

                for batch_idx, data in enumerate(train_dataloader):
                    loss = self.train_batch(batch_idx, data, _tqdm)

            self.logger.info(f"epoch: {epoch+1}/{self.epochs}, lr: {self.scheduler.get_last_lr()},  train loss: {loss:.4f}")
            
            self.scheduler.step()

            # save checkpoint
            self.save_checkpoint(epoch)

            # evalution
            self.evaluation(epoch)

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

        return loss.item()

    def evaluation(self, epoch):
        self.model.eval()

        self.logger.info(f"epoch: {epoch+1}/{self.epochs}, start validation...")

        en_texts = [
            "new Questions Over California Water Project",
            "I had no idea it was a medal but my performance was the best I could have done , that is why I was so happy , that all the training and hard work had paid off .",
        ]
        for en_text in en_texts:
            zh_text = self.translate(en_text)
            self.logger.info("-" * 100)
            self.logger.info(f"[English]: {en_text}")
            self.logger.info(f"[Chinese]: {zh_text}")

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
    trainer = Trainer(config, data_dict)
    trainer.train()
