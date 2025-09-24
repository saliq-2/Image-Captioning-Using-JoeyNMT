from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn

from generation import generate_caption_from_features, tokens_to_caption
from metrics import CaptionMetrics


def validate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, captions in val_loader:
            images = images.to(device)
            captions = captions.to(device)
            inputs = captions[:, :-1]
            targets = captions[:, 1:]
            batch_size, seq_len = inputs.size()
            trg_mask = torch.ones((batch_size, 1, seq_len), dtype=torch.bool, device=inputs.device)
            outputs = model(images, inputs, trg_mask=trg_mask)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs.reshape(-1, len(model.tgt_embedding.weight)), targets.reshape(-1))
            total_loss += loss.item()
    return total_loss / len(val_loader)


def evaluate_batch_captions(model, data_loader, vocab, device, max_samples=100):
    model.eval()
    metrics_calculator = CaptionMetrics()
    all_metrics = defaultdict(list)
    samples_processed = 0
    with torch.no_grad():
        for images, captions in data_loader:
            if samples_processed >= max_samples:
                break
            images = images.to(device)
            batch_size = images.size(0)
            for i in range(batch_size):
                if samples_processed >= max_samples:
                    break
                single_image = images[i:i + 1]
                features = model.encoder(single_image)
                generated_caption = generate_caption_from_features(model, features, vocab, device=device)
                reference_caption = tokens_to_caption(captions[i].cpu().numpy(), vocab)
                if reference_caption.strip() and generated_caption.strip():
                    metrics = metrics_calculator.calculate_all_metrics(reference_caption, generated_caption)
                    for metric_name, score in metrics.items():
                        all_metrics[metric_name].append(score)
                samples_processed += 1
    avg_metrics = {}
    for metric_name, scores in all_metrics.items():
        avg_metrics[metric_name] = np.mean(scores) if scores else 0.0
    return avg_metrics


def evaluate_model_comprehensive(model, test_loader, vocab, device, save_results=True):
    print("\n" + "=" * 60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 60)
    print("\nEvaluating on validation set...")
    metrics = evaluate_batch_captions(model, test_loader, vocab, device, max_samples=200)
    print(f"\nOverall Performance:")
    print(f"  BLEU Score: {metrics.get('bleu', 0.0):.4f}")
    print(f"  METEOR Score: {metrics.get('meteor', 0.0):.4f}")
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_index)
    val_loss = validate_model(model, test_loader, criterion, device)
    val_perplexity = np.exp(min(val_loss, 10))
    print(f"  Validation Loss: {val_loss:.4f}")
    print(f"  Validation Perplexity: {val_perplexity:.2f}")
    if save_results:
        import pickle
        results = {
            'metrics': metrics,
            'validation_loss': val_loss,
            'validation_perplexity': val_perplexity,
        }
        with open('evaluation_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        print(f"\nDetailed results saved to: evaluation_results.pkl")
    return metrics


