import torch
from PIL import Image
from torchvision import transforms

from data import ToTensor


@torch.no_grad()
def generate_caption_from_features(model, features, vocab, max_len=20, device="cpu"):
    model.eval()
    generated_ids = [vocab.bos_index]
    for _ in range(max_len):
        trg = torch.tensor([generated_ids], dtype=torch.long, device=device)
        trg_embed = model.tgt_embedding(trg)
        unroll_steps = trg.size(1)
        trg_mask = torch.ones((1, 1, trg.size(1)), dtype=torch.bool, device=device)
        output = model.decoder(
            trg_embed=trg_embed,
            encoder_output=features,
            encoder_hidden=None,
            src_mask=None,
            unroll_steps=unroll_steps,
            hidden=None,
            trg_mask=trg_mask,
        )
        next_id = output[0][0, -1].argmax(-1).item()
        generated_ids.append(next_id)
        if next_id == vocab.eos_index:
            break
    words = []
    for idx in generated_ids[1:]:
        if idx == vocab.eos_index:
            break
        try:
            words.append(vocab._itos[idx])
        except Exception:
            words.append('<unk>')
    return " ".join(words)


@torch.no_grad()
def generate_caption(model, image, vocab, max_len=20, device="cpu"):
    model.eval()
    if isinstance(image, Image.Image):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = transform(image).unsqueeze(0).to(device)
    features = model.encoder(image)
    return generate_caption_from_features(model, features, vocab, max_len, device)


def tokens_to_caption(token_ids, vocab):
    words = []
    for token_id in token_ids:
        if token_id == vocab.eos_index:
            break
        if token_id not in [vocab.pad_index, vocab.bos_index]:
            try:
                words.append(vocab._itos[token_id])
            except Exception:
                words.append('<unk>')
    return " ".join(words)


