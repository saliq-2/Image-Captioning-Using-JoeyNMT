import torch
import torch.nn as nn
from types import SimpleNamespace
from torchvision import models

from joeynmt.decoders import TransformerDecoder


class VGGEncoder(nn.Module):
    def __init__(self, encoded_size=512, fine_tune=False):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg.features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Linear(512 * 7 * 7, encoded_size)
        self.dropout = nn.Dropout(0.4)
        self.bn = nn.BatchNorm1d(encoded_size)
        if not fine_tune:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn(x)
        return x.unsqueeze(1)


class ResNetEncoder(nn.Module):
    def __init__(self, encoded_size=512, fine_tune=False):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(resnet.fc.in_features, encoded_size)
        self.dropout = nn.Dropout(0.4)
        self.bn = nn.BatchNorm1d(encoded_size)
        if not fine_tune:
            for param in self.resnet.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.resnet(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn(x)
        return x.unsqueeze(1)


class Image2Caption(nn.Module):
    def __init__(self, encoder, decoder, tgt_embedding):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tgt_embedding = tgt_embedding

    def forward(self, images, captions, src_mask=None, trg_mask=None):
        features = self.encoder(images)
        trg_embed = self.tgt_embedding(captions)
        unroll_steps = captions.size(1)
        outputs = self.decoder(
            trg_embed=trg_embed,
            encoder_output=features,
            encoder_hidden=None,
            src_mask=src_mask,
            unroll_steps=unroll_steps,
            hidden=None,
            trg_mask=trg_mask,
        )
        return outputs


def create_model(vocab, device, encoder: str = "vgg16", encoded_size: int = 512, fine_tune: bool = True):
    decoder_config = {
        "num_layers": 2,
        "num_heads": 4,
        "hidden_size": 512,
        "ff_size": 2048,
        "dropout": 0.1,
        "embeddings": {
            "embedding_dim": 512,
            "vocab_size": len(vocab),
            "scale": True,
            "dropout": 0.1,
            "padding_idx": vocab.pad_index,
        },
    }
    decoder = TransformerDecoder(
        cfg=SimpleNamespace(**decoder_config),
        vocab_size=len(vocab),
        pad_index=vocab.pad_index,
        bos_index=vocab.bos_index,
        eos_index=vocab.eos_index,
    ).to(device)
    tgt_embedding = nn.Embedding(len(vocab), 512).to(device)
    if encoder.lower() == "resnet50" or encoder.lower() == "resnet":
        enc = ResNetEncoder(encoded_size=encoded_size, fine_tune=fine_tune).to(device)
    elif encoder.lower() == "vgg16" or encoder.lower() == "vgg":
        enc = VGGEncoder(encoded_size=encoded_size, fine_tune=fine_tune).to(device)
    else:
        raise ValueError(f"Unknown encoder '{encoder}'. Supported: 'vgg16', 'resnet50'")
    model = Image2Caption(enc, decoder, tgt_embedding).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    return model


