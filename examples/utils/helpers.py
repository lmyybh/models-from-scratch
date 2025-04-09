import os
import sys
import yaml
import random
import numpy as np
import torch
import logging
import json


def read_yaml(yaml_file):
    with open(yaml_file, "r") as f:
        content = yaml.safe_load(f)

    return content


def init_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init_logger(log_file, use_stdout=True):
    formatter = logging.Formatter("[%(asctime)s %(levelname)s]: %(message)s")
    if os.path.isfile(log_file):
        os.unlink(log_file)  # delete files

    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(filename=log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if use_stdout:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def dict2str(dict, indent=2):
    return json.dumps(dict, indent=indent)
