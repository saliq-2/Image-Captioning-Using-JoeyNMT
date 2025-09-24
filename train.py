import numpy as np
import torch

from evaluation import validate_model, evaluate_batch_captions


def train_with_comprehensive_metrics(model, train_loader, val_loader, criterion, optimizer, scheduler,
                                     vocab, device, num_epochs=20, eval_frequency=2, save_path='best_model.pth'):
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_perplexity': [],
        'val_perplexity': [],
        'val_bleu': [],
        'val_meteor': [],
        'learning_rate': [],
    }
    best_val_loss = float('inf')
    print("Starting enhanced training with comprehensive metrics...")
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for i, (images, captions) in enumerate(train_loader):
            images = images.to(device)
            captions = captions.to(device)
            inputs = captions[:, :-1]
            targets = captions[:, 1:]
            batch_size, seq_len = inputs.size()
            trg_mask = torch.ones((batch_size, 1, seq_len), dtype=torch.bool, device=inputs.device)
            optimizer.zero_grad()
            outputs = model(images, inputs, trg_mask=trg_mask)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs.reshape(-1, len(vocab)), targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
            if (i + 1) % 50 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        avg_train_loss = total_train_loss / len(train_loader)
        train_perplexity = np.exp(min(avg_train_loss, 10))
        avg_val_loss = validate_model(model, val_loader, criterion, device)
        val_perplexity = np.exp(min(avg_val_loss, 10))
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_perplexity'].append(train_perplexity)
        history['val_perplexity'].append(val_perplexity)
        history['learning_rate'].append(current_lr)
        if (epoch + 1) % eval_frequency == 0 or epoch == num_epochs - 1:
            print(f"Evaluating comprehensive metrics for epoch {epoch + 1}...")
            val_metrics = evaluate_batch_captions(model, val_loader, vocab, device, max_samples=50)
            history['val_bleu'].append(val_metrics.get('bleu', 0.0))
            history['val_meteor'].append(val_metrics.get('meteor', 0.0))
            print(f"Epoch [{epoch + 1}/{num_epochs}] Comprehensive Results:")
            print(f"  Train Loss: {avg_train_loss:.4f} | Train PPL: {train_perplexity:.2f}")
            print(f"  Val Loss: {avg_val_loss:.4f} | Val PPL: {val_perplexity:.2f}")
            print(f"  BLEU: {val_metrics.get('bleu', 0.0):.4f}")
            print(f"  METEOR: {val_metrics.get('meteor', 0.0):.4f}")
            print(f"  Learning Rate: {current_lr:.2e}\n")
        else:
            if history['val_bleu']:
                history['val_bleu'].append(history['val_bleu'][-1])
                history['val_meteor'].append(history['val_meteor'][-1])
            else:
                history['val_bleu'].append(0.0)
                history['val_meteor'].append(0.0)
            print(f"Epoch [{epoch + 1}/{num_epochs}]")
            print(f"  Train Loss: {avg_train_loss:.4f} | Train PPL: {train_perplexity:.2f}")
            print(f"  Val Loss: {avg_val_loss:.4f} | Val PPL: {val_perplexity:.2f}")
            print(f"  Learning Rate: {current_lr:.2e}\n")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'history': history,
            }, save_path)
    return history


