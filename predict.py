import argparse

from PIL import Image

from config import get_device
from inference import load_trained_model
from generation import generate_caption


def main():
    parser = argparse.ArgumentParser(description="Generate a caption for an image using a trained checkpoint.")
    parser.add_argument("image", type=str, help="Path to the input image")
    parser.add_argument("checkpoint", type=str, help="Path to the trained checkpoint .pth file")
    parser.add_argument("--max_len", type=int, default=20, help="Maximum caption length")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    model, vocab, _ = load_trained_model(args.checkpoint, device)

    img = Image.open(args.image).convert("RGB")
    caption = generate_caption(model, img, vocab, max_len=args.max_len, device=device)
    print(caption)


if __name__ == "__main__":
    main()


