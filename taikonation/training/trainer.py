import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
import yaml
import argparse
import json

from taikonation.data.dataset import get_transformer_data_loaders, DIFFICULTY_MAP
from taikonation.experiments.experiment_tracker import ExperimentTracker
from taikonation.models.transformer import TaikoTransformer
from taikonation.data.tokenization import TaikoTokenizer
from taikonation.utils.seed import set_seed
from datetime import datetime
from torch.optim.lr_scheduler import LambdaLR
import math
from collections import defaultdict
import numpy as np
import shutil
from pathlib import Path

class CheckpointManager:
    """Manage model checkpoints with configurable retention policies"""
    def __init__(self, checkpoint_dir, keep_best_n=3, keep_last_n=2, save_every_n_epochs=5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_best_n = keep_best_n
        self.keep_last_n = keep_last_n
        self.save_every_n_epochs = save_every_n_epochs
        self.best_checkpoints = []
        self.last_checkpoints = []
        self.periodic_checkpoints = []

    def save_checkpoint(self, model, optimizer, epoch, val_loss, config, checkpoint_type='regular'):
        """Save a checkpoint with metadata"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if checkpoint_type == 'best':
            filename = f'best_epoch{epoch}_loss{val_loss:.4f}_{timestamp}.pth'
        elif checkpoint_type == 'periodic':
            filename = f'periodic_epoch{epoch}_{timestamp}.pth'
        else:
            filename = f'last_epoch{epoch}_{timestamp}.pth'

        checkpoint_path = self.checkpoint_dir / filename
        model_config = {
            **config['model'],
            'num_genres': model.genre_embedding.num_embeddings,
            'max_sequence_length': int(model.pos_encoder.pe.shape[1]),
        }
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'model_config': model_config,
            'data_config': config['data'],
            'training_config': config['training'],
            'timestamp': timestamp,
            'pytorch_version': str(torch.__version__),
            'checkpoint_type': checkpoint_type
        }

        temp_path = checkpoint_path.with_suffix('.tmp')
        torch.save(checkpoint, temp_path)
        shutil.move(temp_path, checkpoint_path)

        if checkpoint_type == 'best':
            self._update_best_checkpoints(val_loss, checkpoint_path)
        elif checkpoint_type == 'last':
            self._update_last_checkpoints(checkpoint_path)
        elif checkpoint_type == 'periodic':
            self.periodic_checkpoints.append(checkpoint_path)

        print(f"Saved {checkpoint_type} checkpoint: {filename}")
        return checkpoint_path

    def _update_best_checkpoints(self, val_loss, checkpoint_path):
        """Keep only top N best checkpoints"""
        self.best_checkpoints.append((val_loss, checkpoint_path))
        self.best_checkpoints.sort(key=lambda x: x[0])
        if len(self.best_checkpoints) > self.keep_best_n:
            _, path_to_remove = self.best_checkpoints.pop()
            if path_to_remove.exists():
                path_to_remove.unlink()
                print(f"Removed old best checkpoint: {path_to_remove.name}")

    def _update_last_checkpoints(self, checkpoint_path):
        """Keep only last N checkpoints"""
        self.last_checkpoints.append(checkpoint_path)
        if len(self.last_checkpoints) > self.keep_last_n:
            path_to_remove = self.last_checkpoints.pop(0)
            if path_to_remove.exists():
                path_to_remove.unlink()
                print(f"Removed old last checkpoint: {path_to_remove.name}")

    def should_save_periodic(self, epoch):
        """Check if should save periodic checkpoint"""
        return (epoch + 1) % self.save_every_n_epochs == 0

    def get_best_checkpoint(self):
        """Get path to best checkpoint"""
        if self.best_checkpoints:
            return self.best_checkpoints[0][1]
        return None

    def cleanup_all_except_best(self):
        """Remove all checkpoints except the best one"""
        best_path = self.get_best_checkpoint()
        for checkpoint_path in self.checkpoint_dir.glob('*.pth'):
            if checkpoint_path != best_path:
                checkpoint_path.unlink()
        print(f"Cleaned up all checkpoints except best: {best_path.name}")

class MetricsTracker:
    """Track and compute training metrics"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.reset()

    def reset(self):
        self.losses = []
        self.predictions = []
        self.targets = []
        self.note_type_correct = defaultdict(int)
        self.note_type_total = defaultdict(int)

    def update(self, predictions, targets, loss):
        """Update metrics with batch results"""
        self.losses.append(loss.item())
        pred_tokens = predictions.argmax(dim=-1)
        mask = targets != self.tokenizer.vocab["[PAD]"]
        self.predictions.extend(pred_tokens[mask].cpu().numpy())
        self.targets.extend(targets[mask].cpu().numpy())

        for pred, tgt in zip(pred_tokens[mask].cpu().numpy(), targets[mask].cpu().numpy()):
            note_type = self.tokenizer.id_to_token.get(tgt, "[UNK]")
            self.note_type_total[note_type] += 1
            if pred == tgt:
                self.note_type_correct[note_type] += 1

    def compute(self):
        """Compute final metrics"""
        metrics = {}
        metrics['loss'] = float(np.mean(self.losses)) if self.losses else 0.0
        metrics['perplexity'] = float(np.exp(metrics['loss']))
        if len(self.predictions) > 0:
            metrics['accuracy'] = float(np.mean(np.array(self.predictions) == np.array(self.targets)))
        else:
            metrics['accuracy'] = 0.0

        note_accuracies = {}
        for note_type, total in self.note_type_total.items():
            if total > 0:
                acc = self.note_type_correct[note_type] / total
                note_accuracies[note_type] = acc
        metrics['note_accuracies'] = note_accuracies
        metrics['f1_scores'] = self._compute_f1_scores()
        return metrics

    def _compute_f1_scores(self):
        """Compute F1 score for each token type"""
        from sklearn.metrics import f1_score
        if len(self.predictions) == 0:
            return {}

        unique_labels = sorted(set(self.targets))
        f1_macro = f1_score(
            self.targets,
            self.predictions,
            labels=unique_labels,
            average='macro',
            zero_division=0
        )
        f1_per_class = f1_score(
            self.targets,
            self.predictions,
            labels=unique_labels,
            average=None,
            zero_division=0
        )

        return {
            'macro': float(f1_macro),
            'per_token': {
                self.tokenizer.id_to_token.get(label, f"id_{label}"): float(score)
                for label, score in zip(unique_labels, f1_per_class)
            }
        }

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    """
    Create a schedule with linear warmup and cosine annealing.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda)

class EarlyStopping:
    """Early stopping to halt training when validation loss stops improving"""
    def __init__(self, patience=7, min_delta=0.0001, mode='min', verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, epoch, val_metric):
        score = -val_metric if self.mode == 'min' else val_metric

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'  EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0

        return self.early_stop

def load_config(path="config/default.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def create_optimizer_and_scheduler(model, config, total_steps):
    """Create optimizer with weight decay and advanced scheduler"""
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': config['training'].get('weight_decay', 0.01)
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]

    optimizer = optim.AdamW(
        optimizer_grouped_parameters,
        lr=config['training']['learning_rate'],
        betas=(0.9, 0.98),
        eps=1e-6
    )

    num_warmup_steps = int(total_steps * config['training'].get('warmup_ratio', 0.1))
    scheduler_type = config['training'].get('scheduler_type', 'cosine_warmup')

    if scheduler_type == 'cosine_warmup':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps
        )
    elif scheduler_type == 'cosine_restarts':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config['training'].get('restart_period', 10),
            T_mult=2,
            eta_min=config['training']['learning_rate'] * 0.01
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min',
            patience=5,
            factor=0.5,
            min_lr=1e-7
        )

    return optimizer, scheduler

def train_fold(config, fold_idx, *, total_folds=1, progress_callback=None):
    run_name = f"fold_{fold_idx + 1}"
    use_wandb = bool(config['training'].get('use_wandb', False))
    if use_wandb:
        wandb.init(
            project="TaikoNation-Genre-Conditioning",
            config=config,
            name=run_name,
            reinit=True,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Fold {fold_idx + 1}/{total_folds} on {device} ---")
    set_seed(config['training'].get('seed', 42) + fold_idx)

    try:
        with ExperimentTracker(f"train_fold_{fold_idx}", config) as tracker:
            train_loader, val_loader, tokenizer, genre_vocab = get_transformer_data_loaders(config, fold_idx)
            if train_loader is None or val_loader is None:
                raise RuntimeError("Could not build training/validation loaders")

            # Persist training-fold normalization with the checkpoint so
            # research inference sees exactly the same feature distribution.
            train_dataset = train_loader.dataset
            if train_dataset.feature_mean is not None and train_dataset.feature_std is not None:
                config['data']['feature_mean'] = train_dataset.feature_mean.tolist()
                config['data']['feature_std'] = train_dataset.feature_std.tolist()

            model_config = dict(config['model'])
            model_config.setdefault(
                'max_sequence_length',
                int(config['data']['max_sequence_length']),
            )
            model = TaikoTransformer(
                vocab_size=tokenizer.vocab_size,
                num_genres=len(genre_vocab),
                num_difficulties=len(DIFFICULTY_MAP),
                **model_config,
            ).to(device)
            if use_wandb:
                wandb.watch(model, log="all")

            criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab["[PAD]"])
            desired_batch_size = int(config['training'].get('effective_batch_size', config['training']['batch_size']))
            actual_batch_size = int(config['training']['batch_size'])
            accumulation_steps = max(1, math.ceil(desired_batch_size / actual_batch_size))
            optimizer_steps_per_epoch = max(1, math.ceil(len(train_loader) / accumulation_steps))
            total_steps = optimizer_steps_per_epoch * int(config['training']['num_epochs'])
            optimizer, scheduler = create_optimizer_and_scheduler(model, config, total_steps)

            use_amp = device.type == "cuda"
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

            def take_optimizer_step():
                previous_scale = scaler.get_scale()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                # GradScaler intentionally skips optimizer.step() on overflow.
                # Do not advance a step-based scheduler when that happens.
                optimizer_stepped = (not use_amp) or scaler.get_scale() >= previous_scale
                if optimizer_stepped and config['training'].get('scheduler_type') == 'cosine_warmup':
                    scheduler.step()

            best_val_loss = float('inf')

            checkpoint_dir = Path(config['training']['save_path']).parent / f'checkpoints_fold{fold_idx+1}'
            checkpoint_manager = CheckpointManager(
                checkpoint_dir,
                keep_best_n=config['training'].get('keep_best_n', 3),
                keep_last_n=config['training'].get('keep_last_n', 2),
                save_every_n_epochs=config['training'].get('save_every_n_epochs', 5)
            )

            early_stopping = EarlyStopping(
                patience=config['training'].get('early_stopping_patience', 10),
                min_delta=config['training'].get('early_stopping_delta', 1e-4),
                mode='min',
                verbose=True
            )

            print(f"Using gradient accumulation: {accumulation_steps} steps")
            print(f"Effective batch size: {actual_batch_size * accumulation_steps}")

            train_results = {}
            val_results = {}
            num_epochs = int(config['training']['num_epochs'])

            for epoch in range(num_epochs):
                model.train()
                train_metrics = MetricsTracker(tokenizer)
                optimizer.zero_grad(set_to_none=True)
                batches_seen = 0

                for batch_idx, batch in enumerate(train_loader):
                    if not batch:
                        continue
                    batches_seen += 1
                    encoder_input = batch["encoder_input"].to(device, non_blocking=True)
                    decoder_input = batch["decoder_input"].to(device, non_blocking=True)
                    target = batch["target"].to(device, non_blocking=True)
                    padding_mask = batch["padding_mask"].to(device, non_blocking=True)
                    genre_id = batch["genre_id"].to(device, non_blocking=True)
                    difficulty_id = batch["difficulty"].to(device, non_blocking=True)

                    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                        output = model(
                            encoder_input,
                            decoder_input,
                            genre_id,
                            difficulty_id,
                            padding_mask=padding_mask,
                        )
                        raw_loss = criterion(
                            output.reshape(-1, tokenizer.vocab_size),
                            target.reshape(-1),
                        )
                        loss = raw_loss / accumulation_steps

                    scaler.scale(loss).backward()
                    train_metrics.update(output, target, raw_loss)

                    if batches_seen % accumulation_steps == 0:
                        take_optimizer_step()

                if batches_seen == 0:
                    raise RuntimeError("Training loader produced no valid batches")

                if batches_seen % accumulation_steps != 0:
                    take_optimizer_step()

                model.eval()
                val_metrics = MetricsTracker(tokenizer)
                val_batches = 0
                with torch.no_grad():
                    for batch in val_loader:
                        if not batch:
                            continue
                        val_batches += 1
                        encoder_input = batch["encoder_input"].to(device, non_blocking=True)
                        decoder_input = batch["decoder_input"].to(device, non_blocking=True)
                        target = batch["target"].to(device, non_blocking=True)
                        padding_mask = batch["padding_mask"].to(device, non_blocking=True)
                        genre_id = batch["genre_id"].to(device, non_blocking=True)
                        difficulty_id = batch["difficulty"].to(device, non_blocking=True)

                        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                            output = model(
                                encoder_input,
                                decoder_input,
                                genre_id,
                                difficulty_id,
                                padding_mask=padding_mask,
                            )
                            loss = criterion(
                                output.reshape(-1, tokenizer.vocab_size),
                                target.reshape(-1),
                            )
                        val_metrics.update(output, target, loss)

                if val_batches == 0:
                    raise RuntimeError("Validation loader produced no valid batches")

                train_results = train_metrics.compute()
                val_results = val_metrics.compute()

                scheduler_type = config['training'].get('scheduler_type')
                if scheduler_type == 'cosine_restarts':
                    scheduler.step(epoch + 1)
                elif scheduler_type not in {'cosine_warmup'}:
                    scheduler.step(val_results['loss'])

                log_payload = {
                    'epoch': epoch,
                    'train/loss': train_results['loss'],
                    'train/accuracy': train_results['accuracy'],
                    'train/perplexity': train_results['perplexity'],
                    'val/loss': val_results['loss'],
                    'val/accuracy': val_results['accuracy'],
                    'val/perplexity': val_results['perplexity'],
                    'val/f1_macro': val_results.get('f1_scores', {}).get('macro', 0.0),
                    'learning_rate': optimizer.param_groups[0]['lr'],
                }
                if use_wandb:
                    wandb.log(log_payload)
                    for note_type, acc in val_results['note_accuracies'].items():
                        wandb.log({f'val/accuracy_{note_type}': acc})

                print(f"Epoch {epoch+1}/{num_epochs}")
                print(
                    f"  Train - Loss: {train_results['loss']:.4f}, "
                    f"Acc: {train_results['accuracy']:.4f}, "
                    f"PPL: {train_results['perplexity']:.2f}"
                )
                print(
                    f"  Val   - Loss: {val_results['loss']:.4f}, "
                    f"Acc: {val_results['accuracy']:.4f}, "
                    f"PPL: {val_results['perplexity']:.2f}"
                )

                checkpoint_manager.save_checkpoint(
                    model, optimizer, epoch, val_results['loss'],
                    config, checkpoint_type='last'
                )

                if val_results['loss'] < best_val_loss:
                    best_val_loss = val_results['loss']
                    checkpoint_manager.save_checkpoint(
                        model, optimizer, epoch, val_results['loss'],
                        config, checkpoint_type='best'
                    )
                    print("  ✓ New best model saved!")

                if checkpoint_manager.should_save_periodic(epoch):
                    checkpoint_manager.save_checkpoint(
                        model, optimizer, epoch, val_results['loss'],
                        config, checkpoint_type='periodic'
                    )

                if progress_callback is not None:
                    progress_callback({
                        'fold': fold_idx + 1,
                        'folds': total_folds,
                        'epoch': epoch + 1,
                        'epochs': num_epochs,
                        'train': train_results,
                        'val': val_results,
                    })

                if early_stopping(epoch, val_results['loss']):
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    print(
                        f"Best epoch was {early_stopping.best_epoch+1} "
                        f"with val_loss={-early_stopping.best_score:.4f}"
                    )
                    tracker.log_metric("best_epoch", early_stopping.best_epoch + 1)
                    tracker.log_metric("best_val_loss", float(-early_stopping.best_score))
                    break

            best_path = checkpoint_manager.get_best_checkpoint()
            if best_path is None:
                raise RuntimeError("Training completed without producing a best checkpoint")

            if config['training'].get('cleanup_checkpoints', False):
                checkpoint_manager.cleanup_all_except_best()

            configured_output = Path(config['training']['save_path'])
            configured_output.parent.mkdir(parents=True, exist_ok=True)
            if total_folds == 1:
                export_path = configured_output
            else:
                export_path = configured_output.with_name(
                    f"{configured_output.stem}_fold{fold_idx+1}{configured_output.suffix}"
                )
            shutil.copy2(best_path, export_path)

            print(f"--- Finished Fold {fold_idx + 1}; best checkpoint: {export_path} ---")
            tracker.log_metrics({
                "final_train_results": train_results,
                "final_val_results": val_results,
                "best_checkpoint": str(export_path),
            })
            tracker.save_artifact(export_path)

            genre_vocab_path = configured_output.parent / "genre_vocab.json"
            with open(genre_vocab_path, 'w') as handle:
                json.dump(genre_vocab, handle, indent=2)
            print(f"Genre vocabulary saved to {genre_vocab_path}")

            return {
                "fold": fold_idx + 1,
                "checkpoint": str(export_path),
                "train": train_results,
                "val": val_results,
            }
    finally:
        if use_wandb:
            wandb.finish()


def main(config, *, progress_callback=None):
    os.environ.setdefault("WANDB_MODE", "offline")
    num_folds = 1 if config.get('dry_run') else int(config['training']['k_folds'])
    results = []
    for i in range(num_folds):
        results.append(
            train_fold(
                config,
                i,
                total_folds=num_folds,
                progress_callback=progress_callback,
            )
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Taiko Transformer model with genre conditioning.")
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to config file.')
    args = parser.parse_args()
    config = load_config(args.config)
    main(config)