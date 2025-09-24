import json
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from joeynmt.vocabulary import Vocabulary

from config import joeynmt_special_symbols


class ToTensor(object):
    def __call__(self, pic):
        if not isinstance(pic, Image.Image):
            raise TypeError(f"pic should be PIL Image. Got {type(pic)}")
        pic = pic.convert("RGB")
        w, h = pic.size
        img_bytes = pic.tobytes()
        img_tensor = torch.ByteTensor(list(img_bytes)).view(h, w, 3).permute(2, 0, 1).float()
        img_tensor /= 255.0
        return img_tensor


class FlickrDataset(Dataset):
    def __init__(self, captions_json, images_dir, vocab, max_len=50):
        self.images_dir = Path(images_dir)
        self.vocab = vocab
        self.max_len = max_len
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        with open(captions_json, "r") as f:
            self.captions_data = json.load(f)
        self.img_caption_pairs = []
        for img_name, captions in self.captions_data.items():
            for caption in captions:
                self.img_caption_pairs.append((img_name, caption))

    def __len__(self):
        return len(self.img_caption_pairs)

    def __getitem__(self, idx):
        img_name, caption = self.img_caption_pairs[idx]
        img_path = self.images_dir / img_name
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        tokens = caption.lower().strip().split()
        tokens = tokens[: self.max_len - 2]
        token_ids, _, _ = self.vocab.sentences_to_ids([tokens], bos=True, eos=True)
        token_ids = torch.tensor(token_ids[0], dtype=torch.long)
        return image, token_ids


def build_vocabulary(captions_file, min_freq=3):
    try:
        with open(captions_file) as f:
            captions_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Captions file not found at '{captions_file}'. Please update the path.")
        return None
    all_captions = [cap for caps in captions_data.values() for cap in caps]
    tokens = [word for cap in all_captions for word in cap.lower().split()]
    token_counts = Counter(tokens)
    filtered_tokens = [token for token, count in token_counts.items() if count >= min_freq]
    vocab = Vocabulary(filtered_tokens, joeynmt_special_symbols())
    print(f"Vocab size: {len(vocab)} (filtered from {len(token_counts)} unique tokens)")
    return vocab


def custom_collate_fn(batch, vocab):
    images, captions = zip(*batch)
    images = torch.stack(images, 0)
    max_len = max([len(c) for c in captions])
    padded_captions = torch.full((len(captions), max_len), vocab.pad_index, dtype=torch.long)
    for i, caption in enumerate(captions):
        padded_captions[i, : len(caption)] = caption
    return images, padded_captions


