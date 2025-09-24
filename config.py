import os
import random
from types import SimpleNamespace

import numpy as np
import torch


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_global_seeds(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def default_config() -> dict:
    return {
        'captions_file': "/home/s124z/Documents/joeynmtfinal/archive/flickrr2000/captions_2000.json",
        'images_dir': "/home/s124z/Documents/joeynmtfinal/archive/flickrr2000/flickr2000",
        'encoder': 'vgg16',  # 'vgg16' or 'resnet50'
        'encoded_size': 512,
        'fine_tune_encoder': True,
        'min_freq': 3,
        'max_len': 30,
        'batch_size': 16,
        'num_epochs': 2,
        'learning_rate': 5e-5,
        'weight_decay': 1e-3,
        'eval_frequency': 2,
        'patience': 12,
        'save_path': 'best_image_captioning_model.pth'
    }


def joeynmt_special_symbols() -> SimpleNamespace:
    return SimpleNamespace(
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<sos>",
        eos_token="<eos>",
        sep_token=None,
        unk_id=0,
        pad_id=1,
        bos_id=2,
        eos_id=3,
        lang_tags=[]
    )


