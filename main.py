import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from config import default_config, get_device, set_global_seeds
from data import FlickrDataset, build_vocabulary, custom_collate_fn
from models import create_model
from train import train_with_comprehensive_metrics
from visualization import plot_training_curves, print_final_metrics_summary
from evaluation import evaluate_model_comprehensive


def run_training_pipeline():
    CONFIG = default_config()
    device = get_device()
    print(f"Using device: {device}")
    set_global_seeds(42)
    print("=" * 60)
    print("IMAGE CAPTIONING MODEL TRAINING")
    print("=" * 60)
    print("\n1. Building vocabulary...")
    vocab = build_vocabulary(CONFIG['captions_file'], CONFIG['min_freq'])
    if vocab is None:
        return
    print("\n2. Creating datasets and dataloaders...")
    dataset = FlickrDataset(
        captions_json=CONFIG['captions_file'],
        images_dir=CONFIG['images_dir'],
        vocab=vocab,
        max_len=CONFIG['max_len'],
    )
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    def collate_fn(batch):
        return custom_collate_fn(batch, vocab)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=collate_fn)
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print("\n3. Creating model...")
    model = create_model(
        vocab,
        device,
        encoder=CONFIG['encoder'],
        encoded_size=CONFIG['encoded_size'],
        fine_tune=CONFIG['fine_tune_encoder'],
    )
    print("\n4. Setting up training components...")
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_index)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.7)
    print("\n5. Starting training...")
    history = train_with_comprehensive_metrics(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        vocab=vocab,
        device=device,
        num_epochs=CONFIG['num_epochs'],
        eval_frequency=CONFIG['eval_frequency'],
        save_path=CONFIG['save_path'],
    )
    print("\n6. Generating results...")
    plot_training_curves(history, 'comprehensive_training_curves.png')
    print_final_metrics_summary(history)
    print("\n7. Optional: Comprehensive evaluation...")
    # evaluate_model_comprehensive(model, val_loader, vocab, device)
    print(f"\n8. Training completed! Best model saved as: {CONFIG['save_path']}")
    print("Training curves saved as: comprehensive_training_curves.png")
    return model, history, vocab


if __name__ == "__main__":
    run_training_pipeline()


