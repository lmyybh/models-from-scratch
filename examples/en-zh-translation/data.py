import yaml
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


class NewsCommentaryDataset(Dataset):
    def __init__(self, en_file, zh_file) -> None:
        self.data = self.read_data(en_file, zh_file)

    def read_data(self, en_file, zh_file):
        with open(en_file, "r") as f:
            en_lines = [line.rstrip() for line in f.readlines()]

        with open(zh_file, "r") as f:
            zh_lines = [line.rstrip() for line in f.readlines()]

        return list(zip(en_lines, zh_lines))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        en_text, zh_text = self.data[index]

        return en_text, zh_text


class CollactorWithPadding:
    def __init__(
        self,
        en_tokenizer,
        zh_tokenizer,
        padding=True,
        max_length=512,
        truncation=True,
        reture_tensors="pt",
    ) -> None:
        self.en_tokenizer = en_tokenizer
        self.zh_tokenizer = zh_tokenizer
        self.padding = padding
        self.max_length = max_length
        self.truncation = truncation
        self.reture_tensors = reture_tensors
        
    def encode_english(self, batch_en):
        batch_en_tokens = self.en_tokenizer.batch_encode_plus(
            batch_en,
            padding=self.padding,
            max_length=self.max_length,
            truncation=self.truncation,
            return_tensors=self.reture_tensors,
        )
        batch_en_tokens["attention_mask"] = ~batch_en_tokens["attention_mask"].to(bool)
        
        return batch_en_tokens
        
    def encode_chinese(self, batch_zh):
        batch_zh_tokens = self.zh_tokenizer.batch_encode_plus(
            batch_zh,
            padding=self.padding,
            max_length=self.max_length,
            truncation=self.truncation,
            return_tensors=self.reture_tensors,
        )
        batch_zh_tokens["attention_mask"] = ~batch_zh_tokens["attention_mask"].to(bool)
        
        return batch_zh_tokens
        

    def __call__(self, batch):
        batch_en, batch_zh = list(zip(*batch))

        batch_en_tokens = self.encode_english(batch_en)
        batch_zh_tokens = self.encode_chinese(batch_zh)

        return {"en": batch_en_tokens, "zh": batch_zh_tokens}


# load config
with open("./config.yaml", "r") as f:
    config = yaml.safe_load(f)

# load tokenizer
en_tokenizer = BertTokenizer.from_pretrained(config["tokenizer"]["en"])
zh_tokenizer = BertTokenizer.from_pretrained(config["tokenizer"]["zh"])

# init collator
collactor = CollactorWithPadding(
    en_tokenizer=en_tokenizer,
    zh_tokenizer=zh_tokenizer,
    padding=config["tokenizer"]["padding"],
    max_length=config["tokenizer"]["max_length"],
    truncation=config["tokenizer"]["truncation"],
    reture_tensors=config["tokenizer"]["reture_tensors"],
)

# set dataloader
train_dataloader = DataLoader(
    NewsCommentaryDataset(
        en_file=config["dataset"]["train"]["en"],
        zh_file=config["dataset"]["train"]["zh"],
    ),
    batch_size=config["train"]["batch_size"],
    shuffle=True,
    num_workers=config["train"]["num_workers"],
    collate_fn=collactor,
)
