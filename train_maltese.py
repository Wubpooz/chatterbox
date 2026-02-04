#!/usr/bin/env python3
"""
Training Script Template for Maltese Language Fine-Tuning

This script provides a starting point for fine-tuning the Chatterbox multilingual
TTS model to add proper Maltese language support.

Requirements:
- Maltese dataset with audio + transcripts
- GPU with 16GB+ VRAM (V100, A100, or RTX 3090/4090)
- Additional Arabic and Italian data to prevent forgetting

Usage:
    python train_maltese.py --config config.yaml --data_dir /path/to/data
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultilingualTTSDataset(Dataset):
    """
    Dataset for multilingual TTS training.
    
    Expected data format:
    {
        "audio": "path/to/audio.wav",
        "text": "Bonġu! Kif int illum?",
        "language": "mt",
        "speaker_id": "speaker_001"
    }
    """
    
    def __init__(self, data_list: List[Dict], tokenizer, speech_tokenizer):
        self.data = data_list
        self.tokenizer = tokenizer
        self.speech_tokenizer = speech_tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize text
        text_tokens = self.tokenizer.text_to_tokens(
            item['text'],
            language_id=item['language']
        )
        
        # Process audio and extract speech tokens
        # (This is a placeholder - actual implementation depends on your data pipeline)
        # speech_tokens = self.speech_tokenizer.encode(item['audio'])
        
        return {
            'text': item['text'],
            'text_tokens': text_tokens,
            'language': item['language'],
            'audio_path': item['audio'],
            # 'speech_tokens': speech_tokens,
        }


class MalteseTrainer:
    """Trainer for fine-tuning on Maltese language data."""
    
    def __init__(
        self,
        model: ChatterboxMultilingualTTS,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Setup training
        self._setup_training()
    
    def _setup_training(self):
        """Configure optimizer, scheduler, and training parameters."""
        
        # Freeze speech encoder and decoder (as noted in problem statement)
        logger.info("Freezing speech encoder and decoder...")
        self.model.ve.requires_grad_(False)
        self.model.s3gen.requires_grad_(False)
        
        # Count trainable parameters
        trainable = sum(p.numel() for p in self.model.t3.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable parameters: {trainable / 1e6:.2f}M / {total / 1e6:.2f}M")
        
        # Optimizer
        self.optimizer = AdamW(
            [p for p in self.model.t3.parameters() if p.requires_grad],
            lr=self.config.get('learning_rate', 1e-5),
            weight_decay=self.config.get('weight_decay', 0.01),
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.get('max_steps', 10000)
        )
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        max_steps = self.config.get('max_steps', 10000)
        log_interval = self.config.get('log_interval', 100)
        val_interval = self.config.get('val_interval', 1000)
        save_interval = self.config.get('save_interval', 1000)
        
        self.model.t3.train()
        
        while self.global_step < max_steps:
            for batch in tqdm(self.train_loader, desc=f"Step {self.global_step}"):
                # Forward pass
                # NOTE: This is a simplified version
                # Actual implementation requires proper batch preparation
                # with text_tokens, speech_tokens, lengths, and conditioning
                
                try:
                    loss = self._training_step(batch)
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.t3.parameters(),
                        max_norm=self.config.get('max_grad_norm', 1.0)
                    )
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    # Logging
                    if self.global_step % log_interval == 0:
                        logger.info(
                            f"Step {self.global_step}: "
                            f"Loss={loss.item():.4f}, "
                            f"LR={self.optimizer.param_groups[0]['lr']:.2e}"
                        )
                    
                    # Validation
                    if self.global_step % val_interval == 0:
                        val_loss = self.validate()
                        logger.info(f"Validation loss: {val_loss:.4f}")
                        
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.save_checkpoint('best_model.pt')
                    
                    # Save checkpoint
                    if self.global_step % save_interval == 0:
                        self.save_checkpoint(f'checkpoint_step_{self.global_step}.pt')
                    
                    self.global_step += 1
                    
                    if self.global_step >= max_steps:
                        break
                        
                except Exception as e:
                    logger.error(f"Error in training step {self.global_step}: {e}")
                    continue
        
        logger.info("Training complete!")
    
    def _training_step(self, batch):
        """
        Perform a single training step.
        
        NOTE: This is a placeholder that needs to be implemented based on
        your actual data pipeline and batch structure.
        """
        # TODO: Implement proper batch processing
        # This should call model.t3.loss() with proper arguments:
        # - t3_cond (conditioning from voice encoder + prompt)
        # - text_tokens (tokenized input text)
        # - text_token_lens (lengths of text sequences)
        # - speech_tokens (target speech token sequence)
        # - speech_token_lens (lengths of speech sequences)
        
        raise NotImplementedError(
            "Training step needs to be implemented based on your data pipeline. "
            "See MALTESE_FINETUNING_GUIDE.md for details."
        )
    
    def validate(self):
        """Run validation on held-out data."""
        self.model.t3.eval()
        
        val_losses = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                try:
                    loss = self._training_step(batch)
                    val_losses.append(loss.item())
                except Exception as e:
                    logger.warning(f"Validation batch error: {e}")
                    continue
        
        self.model.t3.train()
        
        return sum(val_losses) / len(val_losses) if val_losses else float('inf')
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        save_path = Path(self.config.get('output_dir', '.')) / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'step': self.global_step,
            't3_state_dict': self.model.t3.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"Saved checkpoint to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Chatterbox for Maltese")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to data directory")
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help="Output directory")
    parser.add_argument('--learning_rate', type=float, default=1e-5, help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    parser.add_argument('--max_steps', type=int, default=10000, help="Maximum training steps")
    parser.add_argument('--device', type=str, default='cuda', help="Device (cuda/cpu)")
    parser.add_argument('--resume', type=str, default=None, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'learning_rate': args.learning_rate,
        'weight_decay': 0.01,
        'max_steps': args.max_steps,
        'max_grad_norm': 1.0,
        'log_interval': 100,
        'val_interval': 1000,
        'save_interval': 1000,
        'output_dir': args.output_dir,
    }
    
    # Load model
    logger.info("Loading pre-trained model...")
    model = ChatterboxMultilingualTTS.from_pretrained(device=args.device)
    
    # Resize embeddings if vocabulary was expanded
    # Uncomment if you've regenerated the tokenizer vocabulary with [mt] token
    # new_vocab_size = 2455  # 2454 + 1 for [mt]
    # model.t3.resize_text_token_embeddings(new_vocab_size)
    
    # Load data
    # TODO: Implement your data loading logic
    logger.info("Loading training data...")
    # train_dataset = MultilingualTTSDataset(...)
    # val_dataset = MultilingualTTSDataset(...)
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # NOTE: You need to implement the data loading based on your dataset format
    raise NotImplementedError(
        "Data loading needs to be implemented. "
        "See MALTESE_FINETUNING_GUIDE.md for dataset format and requirements."
    )
    
    # Initialize trainer
    # trainer = MalteseTrainer(model, train_loader, val_loader, config)
    
    # Resume from checkpoint if specified
    # if args.resume:
    #     checkpoint = torch.load(args.resume)
    #     model.t3.load_state_dict(checkpoint['t3_state_dict'])
    #     trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    #     trainer.global_step = checkpoint['step']
    #     logger.info(f"Resumed from step {trainer.global_step}")
    
    # Train
    # trainer.train()


if __name__ == '__main__':
    main()
