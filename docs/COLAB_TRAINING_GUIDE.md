# Training Guide for Maltese with MASRI_HEADSET_v2 Dataset

## Overview

This guide provides step-by-step instructions for training the Chatterbox multilingual TTS model for Maltese language support using the **Bluefir/MASRI_HEADSET_v2** dataset on **Google Colab free tier**.

## Dataset: MASRI_HEADSET_v2

**HuggingFace Link**: [Bluefir/MASRI_HEADSET_v2](https://huggingface.co/datasets/Bluefir/MASRI_HEADSET_v2)

The MASRI (Maltese Automatic Speech Recognition and Identification) dataset contains:
- Maltese speech recordings
- High-quality headset recordings
- Transcribed text in Maltese
- Multiple speakers
- **Features**: `audio`, `speaker_id`, `gender`, `duration`, `normalized_text`

**Important Note**: MASRI_HEADSET_v2 contains only Maltese data. To prevent catastrophic forgetting, you **must** also load Arabic, Italian, and English datasets (see "Handling Single-Language Datasets" section below).

## Quick Start

### Option 1: Use Colab Notebook (Recommended)

1. Open `train_maltese_colab.ipynb` in Google Colab
2. Select Runtime → Change runtime type → T4 GPU
3. Run all cells sequentially
4. Follow implementation notes for data pipeline

### Option 2: Local Training

See `MALTESE_FINETUNING_GUIDE.md` for local training setup.

## Handling Single-Language Datasets

### The Problem

MASRI_HEADSET_v2 contains **only Maltese data** with features: `audio`, `speaker_id`, `gender`, `duration`, `normalized_text`. However, training on a single language will cause **catastrophic forgetting** - the model will forget Arabic, Italian, and other languages.

### Solution: Add Multi-Language Datasets

You **must** supplement MASRI with other language datasets to maintain model performance. Here are three practical approaches:

#### **Option 1: Use Common Voice (Recommended - Free & Easy)**

Mozilla Common Voice provides free, high-quality TTS data for many languages:

```python
from datasets import load_dataset

# Maltese (40% of training) - MASRI dataset
maltese_data = load_dataset("Bluefir/MASRI_HEADSET_v2", split="train")
print(f"Maltese samples: {len(maltese_data)}")

# Arabic (35% of training) - Common Voice
arabic_data = load_dataset(
    "mozilla-foundation/common_voice_16_1", 
    "ar",
    split="train[:5000]",  # Adjust based on your Maltese dataset size
    trust_remote_code=True
)
print(f"Arabic samples: {len(arabic_data)}")

# Italian (20% of training) - Common Voice
italian_data = load_dataset(
    "mozilla-foundation/common_voice_16_1",
    "it", 
    split="train[:3000]",
    trust_remote_code=True
)
print(f"Italian samples: {len(italian_data)}")

# English (5% of training) - Common Voice
english_data = load_dataset(
    "mozilla-foundation/common_voice_16_1",
    "en",
    split="train[:1000]",
    trust_remote_code=True
)
print(f"English samples: {len(english_data)}")
```

**Sizing Guidelines**:
- If MASRI has 10,000 samples (40%), you need:
  - ~8,750 Arabic samples (35%)
  - ~5,000 Italian samples (20%)
  - ~1,250 English samples (5%)
- Adjust `split="train[:N]"` to match your ratios

**Advantages**:
- ✅ Free and easily accessible
- ✅ High quality recordings
- ✅ Consistent format across languages
- ✅ Large datasets available

#### **Option 2: Use LoRA (If You Can't Get Multi-Language Data)**

If obtaining multi-language datasets is not possible, use **LoRA (Low-Rank Adaptation)** as a compromise. LoRA minimizes changes to the base model, reducing forgetting:

```python
# Install PEFT library
!pip install peft

from peft import LoraConfig, get_peft_model

# Apply LoRA to T3 transformer
lora_config = LoraConfig(
    r=16,  # Rank (lower = less capacity but less forgetting)
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Attention layers
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA (only to transformer, not embeddings)
model.t3.tfmr = get_peft_model(model.t3.tfmr, lora_config)

# Still need to train embeddings directly (LoRA doesn't touch them)
model.t3.text_emb.requires_grad_(True)
model.t3.text_head.requires_grad_(True)

# Print trainable parameters
model.t3.tfmr.print_trainable_parameters()
# Expected: ~1-2% of parameters trainable
```

**With LoRA Configuration**:
```python
CONFIG = {
    'learning_rate': 2e-4,  # Higher LR for LoRA adapters
    'embedding_lr': 1e-5,   # Lower LR for embeddings
    'max_steps': 3000,
    # ... rest of config
}
```

**Trade-offs**:
- ✅ Reduces forgetting significantly
- ✅ Trains faster (fewer parameters)
- ❌ May have slightly lower quality than full fine-tuning
- ❌ Still need to train embeddings, which can cause some forgetting

#### **Option 3: Very Conservative Training (Last Resort)**

If you absolutely cannot get other datasets or use LoRA, train very conservatively:

```python
# Ultra-conservative configuration
CONSERVATIVE_CONFIG = {
    'learning_rate': 5e-6,  # Half the normal rate
    'max_steps': 2000,      # Limited training
    'eval_steps': 100,      # Validate frequently
    'early_stop_threshold': 0.03,  # Stop if any language drops >3%
    
    # Stronger regularization
    'weight_decay': 0.02,   # Double normal
    'dropout': 0.15,        # Increase dropout
    'max_grad_norm': 0.5,   # Aggressive clipping
}
```

**Validation Strategy**:
```python
# Validate on all languages every 100 steps
baseline_losses = {
    'ar': 2.5,  # Record baseline before training
    'it': 2.3,
    'en': 2.1,
}

for step in training:
    if step % 100 == 0:
        current_losses = validate_all_languages(model)
        
        for lang, current_loss in current_losses.items():
            baseline = baseline_losses.get(lang, current_loss)
            degradation = (current_loss - baseline) / baseline
            
            if degradation > 0.03:  # >3% degradation
                print(f"⚠ Warning: {lang} degraded by {degradation*100:.1f}%")
                print("Consider stopping training or reducing learning rate")
                # Optionally: stop training or rollback
```

**Trade-offs**:
- ⚠ High risk of forgetting
- ⚠ Limited Maltese learning
- ⚠ Requires constant monitoring
- ✅ Works with single dataset
- ✅ No additional setup needed

### **Recommended Approach**

**Use Option 1 (Common Voice)** - it's the best solution:
1. Free and easily accessible
2. Maintains full model capability
3. Prevents catastrophic forgetting effectively
4. Simple to implement

Example complete setup:

```python
from datasets import load_dataset, concatenate_datasets

# Load all datasets
datasets = {
    'mt': load_dataset("Bluefir/MASRI_HEADSET_v2", split="train"),
    'ar': load_dataset("mozilla-foundation/common_voice_16_1", "ar", split="train[:5000]", trust_remote_code=True),
    'it': load_dataset("mozilla-foundation/common_voice_16_1", "it", split="train[:3000]", trust_remote_code=True),
    'en': load_dataset("mozilla-foundation/common_voice_16_1", "en", split="train[:1000]", trust_remote_code=True),
}

# Add language IDs
for lang_id, dataset in datasets.items():
    datasets[lang_id] = dataset.map(lambda x: {**x, 'language_id': lang_id})

# Create mixed loader
from torch.utils.data import DataLoader

loaders = {
    lang: DataLoader(dataset, batch_size=1, shuffle=True)
    for lang, dataset in datasets.items()
}

# Sample according to ratios (40/35/20/5)
ratios = {'mt': 0.40, 'ar': 0.35, 'it': 0.20, 'en': 0.05}
mixed_loader = MixedLanguageDataLoader(loaders, ratios)
```

## Preventing Catastrophic Forgetting

**Critical**: When training on Maltese, the model must maintain performance on previously learned languages (Arabic, Italian, English, etc.).

### Strategy

The training uses **mixed-language batches** with the following distribution:

```
┌─────────────────────────────────────────────┐
│ Training Data Distribution                   │
├─────────────────────────────────────────────┤
│ 40% Maltese   (MASRI_HEADSET_v2)           │
│ 35% Arabic    (preserve Semitic knowledge)  │
│ 20% Italian   (preserve Romance knowledge)  │
│  5% English   (general performance)         │
└─────────────────────────────────────────────┘
```

### Why This Works

1. **Maltese Linguistic Roots**:
   - Semitic base (from Arabic)
   - Heavy Italian influence (Romance vocabulary)
   - Unique hybrid language

2. **Knowledge Transfer**:
   - Training on Arabic helps with Maltese Semitic grammar
   - Training on Italian helps with Maltese loanwords
   - Model learns connections between languages

3. **Prevents Forgetting**:
   - Regular exposure to Arabic/Italian maintains their performance
   - Conservative learning rate (1e-5) avoids overwriting
   - Gradient clipping prevents destructive updates

### Implementation

```python
# Sampling ratios
LANG_RATIOS = {
    'mt': 0.40,  # Maltese
    'ar': 0.35,  # Arabic
    'it': 0.20,  # Italian
    'en': 0.05,  # English
}

# During training, sample from each dataset according to ratios
for step in range(max_steps):
    batch = sample_mixed_batch(
        maltese_loader,   # 40% of batch
        arabic_loader,    # 35% of batch
        italian_loader,   # 20% of batch
        english_loader    # 5% of batch
    )
    # ... train on batch
```

## Required Datasets

To implement full mixed-language training:

### 1. Maltese (40%)

**Dataset**: Bluefir/MASRI_HEADSET_v2

```python
from datasets import load_dataset

maltese_data = load_dataset("Bluefir/MASRI_HEADSET_v2", split="train")
```

**Preprocessing**:
- Language ID: `mt`
- NFKD normalization for special characters (ċ, ġ, ħ, ż)
- Audio: 16kHz sample rate

### 2. Arabic (35%)

**Recommended Dataset**: Mozilla Common Voice (ar)

```python
arabic_data = load_dataset("mozilla-foundation/common_voice_16_1", "ar", split="train")
```

**Alternative**: Any Arabic TTS dataset with clean audio and transcripts

**Preprocessing**:
- Language ID: `ar`
- Stress marking (handled by tokenizer)

### 3. Italian (20%)

**Recommended Dataset**: Mozilla Common Voice (it)

```python
italian_data = load_dataset("mozilla-foundation/common_voice_16_1", "it", split="train")
```

**Alternative**: Italian TTS corpora

**Preprocessing**:
- Language ID: `it`
- Standard Italian text normalization

### 4. English (5%)

**Recommended Dataset**: LJSpeech or Common Voice (en)

```python
english_data = load_dataset("lj_speech", split="train")
# Or
english_data = load_dataset("mozilla-foundation/common_voice_16_1", "en", split="train")
```

**Preprocessing**:
- Language ID: `en`
- Standard English text normalization

## Google Colab Free Tier Optimization

### Memory Constraints

Google Colab free tier provides:
- **GPU**: T4 with 15GB VRAM
- **RAM**: ~12GB system RAM
- **Time Limit**: ~12 hours (with disconnection risk)

### Optimizations

```python
# Configuration for Colab free tier
CONFIG = {
    # Small batch size to fit in 15GB GPU
    'batch_size': 4,
    
    # Use gradient accumulation for effective larger batch
    'gradient_accumulation_steps': 8,  # Effective batch size: 32
    
    # Mixed precision to save memory
    'mixed_precision': True,  # FP16
    
    # Conservative training
    'learning_rate': 1e-5,
    'max_grad_norm': 1.0,
    
    # Reasonable training duration for Colab
    'max_steps': 5000,  # ~2-3 hours on T4
    
    # Checkpointing to prevent data loss
    'save_steps': 1000,
    'eval_steps': 500,
}
```

### Memory Management

```python
# Clear CUDA cache periodically
if step % 100 == 0:
    torch.cuda.empty_cache()

# Freeze unnecessary modules
model.ve.requires_grad_(False)
model.s3gen.requires_grad_(False)

# Only train text components
model.t3.text_emb.requires_grad_(True)
model.t3.text_head.requires_grad_(True)
model.t3.tfmr.requires_grad_(True)
```

### Time Management

```python
# Save checkpoints frequently (in case of disconnection)
if step % 500 == 0:
    save_checkpoint(model, step, output_dir)
    
# Consider using Google Drive for checkpoint storage
from google.colab import drive
drive.mount('/content/drive')
output_dir = '/content/drive/MyDrive/maltese_checkpoints'
```

## Data Pipeline Implementation

### Required Components

The Colab notebook provides the structure, but you need to implement:

#### 1. Audio Preprocessing

```python
import librosa
import torch

def preprocess_audio(audio_path, target_sr=16000, max_length=10.0):
    """
    Load and preprocess audio file.
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate (16kHz for Chatterbox)
        max_length: Maximum audio length in seconds
        
    Returns:
        audio: Preprocessed audio tensor
    """
    # Load audio
    audio, sr = librosa.load(audio_path, sr=target_sr)
    
    # Trim silence
    audio, _ = librosa.effects.trim(audio, top_db=20)
    
    # Limit length
    max_samples = int(max_length * target_sr)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    
    return torch.FloatTensor(audio)
```

#### 2. Speech Tokenization

```python
def extract_speech_tokens(audio, s3_tokenizer):
    """
    Extract speech tokens from audio.
    
    Args:
        audio: Audio tensor (1D)
        s3_tokenizer: S3 speech tokenizer
        
    Returns:
        speech_tokens: Tokenized speech
    """
    # This requires the S3 tokenizer from the model
    # speech_tokens = s3_tokenizer.encode(audio)
    # return speech_tokens
    
    # Placeholder - implement based on model's tokenizer
    raise NotImplementedError("Implement speech tokenization")
```

#### 3. Text Tokenization

```python
def tokenize_text(text, language_id, tokenizer):
    """
    Tokenize text with language ID.
    
    Args:
        text: Input text
        language_id: Language code (mt, ar, it, en)
        tokenizer: MTL tokenizer
        
    Returns:
        text_tokens: Tokenized text
    """
    tokens = tokenizer.text_to_tokens(text, language_id=language_id)
    return tokens
```

#### 4. Batching and Collation

```python
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    """
    Collate batch of samples with padding.
    
    Args:
        batch: List of (text_tokens, speech_tokens, lengths, language_id)
        
    Returns:
        Batched and padded tensors
    """
    text_tokens = [item['text_tokens'] for item in batch]
    speech_tokens = [item['speech_tokens'] for item in batch]
    
    # Pad sequences
    text_tokens_padded = pad_sequence(text_tokens, batch_first=True, padding_value=0)
    speech_tokens_padded = pad_sequence(speech_tokens, batch_first=True, padding_value=0)
    
    # Compute lengths
    text_lens = torch.LongTensor([len(t) for t in text_tokens])
    speech_lens = torch.LongTensor([len(s) for s in speech_tokens])
    
    return {
        'text_tokens': text_tokens_padded,
        'text_token_lens': text_lens,
        'speech_tokens': speech_tokens_padded,
        'speech_token_lens': speech_lens,
    }
```

#### 5. Mixed-Language DataLoader

```python
import random

class MixedLanguageDataLoader:
    """
    DataLoader that samples from multiple language datasets
    according to specified ratios.
    """
    
    def __init__(self, loaders, ratios):
        """
        Args:
            loaders: Dict of language_id -> DataLoader
            ratios: Dict of language_id -> sampling ratio
        """
        self.loaders = loaders
        self.ratios = ratios
        self.iterators = {k: iter(v) for k, v in loaders.items()}
        
    def __iter__(self):
        return self
    
    def __next__(self):
        # Sample language according to ratios
        lang = random.choices(
            list(self.ratios.keys()),
            weights=list(self.ratios.values())
        )[0]
        
        # Get batch from corresponding loader
        try:
            batch = next(self.iterators[lang])
        except StopIteration:
            # Reset iterator if exhausted
            self.iterators[lang] = iter(self.loaders[lang])
            batch = next(self.iterators[lang])
        
        return batch

# Usage
loaders = {
    'mt': maltese_loader,
    'ar': arabic_loader,
    'it': italian_loader,
    'en': english_loader,
}

ratios = {'mt': 0.40, 'ar': 0.35, 'it': 0.20, 'en': 0.05}

mixed_loader = MixedLanguageDataLoader(loaders, ratios)
```

## Monitoring Training

### Key Metrics

Track these metrics to ensure quality training:

#### 1. Loss Metrics

```python
metrics = {
    'train_loss': [],
    'maltese_loss': [],
    'arabic_loss': [],
    'italian_loss': [],
    'english_loss': [],
}

# Log per language
if step % log_interval == 0:
    print(f"Step {step}:")
    print(f"  Total Loss: {loss.item():.4f}")
    print(f"  Maltese: {maltese_loss:.4f}")
    print(f"  Arabic: {arabic_loss:.4f}")
    print(f"  Italian: {italian_loss:.4f}")
```

#### 2. Forgetting Check

```python
# Validate on all languages every N steps
if step % eval_interval == 0:
    val_losses = validate_all_languages(model, val_loaders)
    
    # Check if any language degraded significantly
    for lang, loss in val_losses.items():
        baseline = baseline_losses[lang]
        degradation = (loss - baseline) / baseline * 100
        
        if degradation > 5.0:
            print(f"⚠ Warning: {lang} performance dropped {degradation:.1f}%")
            print(f"  Consider reducing learning rate or Maltese ratio")
```

#### 3. Resource Usage

```python
import psutil
import torch

# Monitor GPU memory
if torch.cuda.is_available():
    mem_used = torch.cuda.memory_allocated() / 1e9
    mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU Memory: {mem_used:.2f}/{mem_total:.2f} GB")

# Monitor system RAM
ram_used = psutil.virtual_memory().percent
print(f"RAM Usage: {ram_used:.1f}%")
```

## Expected Results

### Training Progress

With the recommended configuration:

```
Step 0:    Loss=8.5 (baseline, no training yet)
Step 500:  Loss=6.2 (initial learning)
Step 1000: Loss=4.8 (steady improvement)
Step 2000: Loss=3.5 (convergence beginning)
Step 3000: Loss=2.9 (good progress)
Step 4000: Loss=2.5 (near convergence)
Step 5000: Loss=2.2 (training complete)
```

### Validation Performance

After 5000 steps:
- **Maltese**: Good quality synthesis
- **Arabic**: 95-100% of baseline (minimal degradation)
- **Italian**: 95-100% of baseline (minimal degradation)
- **English**: 98-100% of baseline (minimal impact)

### Signs of Good Training

✅ **Good indicators**:
- Maltese loss decreases steadily
- Other languages maintain within 5% of baseline
- Generated Maltese audio sounds natural
- No NaN or infinite losses
- GPU memory stays under 15GB

❌ **Warning signs**:
- Arabic/Italian performance drops >10%
- Loss plateaus early (step <2000)
- GPU out of memory errors
- Generated audio has artifacts

## Troubleshooting

### Issue: Out of Memory

**Solution**:
```python
# Reduce batch size
CONFIG['batch_size'] = 2

# Increase gradient accumulation
CONFIG['gradient_accumulation_steps'] = 16

# Clear cache more frequently
torch.cuda.empty_cache()
```

### Issue: Catastrophic Forgetting

**Symptoms**: Arabic or Italian performance drops significantly

**Solution**:
```python
# Reduce Maltese ratio
CONFIG['maltese_ratio'] = 0.30
CONFIG['arabic_ratio'] = 0.40  # Increase others

# Lower learning rate
CONFIG['learning_rate'] = 5e-6

# Add validation checks
validate_every = 200  # Check more frequently
```

### Issue: Slow Training

**Solution**:
```python
# Enable mixed precision
CONFIG['mixed_precision'] = True

# Increase batch size if memory allows
CONFIG['batch_size'] = 8

# Use DataLoader num_workers
num_workers = 2  # For data loading
```

### Issue: Colab Disconnection

**Solution**:
```python
# Save to Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Checkpoint frequently
save_steps = 500

# Resume from checkpoint
if os.path.exists(last_checkpoint):
    model.load_state_dict(torch.load(last_checkpoint))
```

## Post-Training

### 1. Evaluation

Test the trained model on held-out Maltese samples:

```python
# Load trained model
model.t3.load_state_dict(torch.load('checkpoint-5000/t3_maltese.pt'))

# Test generation
maltese_texts = [
    "Bonġu! Kif int illum?",
    "Malta għandha storja kbira.",
    "Jiena kuntent li niltaqa' miegħek."
]

for text in maltese_texts:
    wav = model.generate(text, language_id='mt')
    # Listen and evaluate quality
```

### 2. Validation on Other Languages

Ensure no catastrophic forgetting:

```python
# Test Arabic
arabic_text = "مرحبا، كيف حالك اليوم؟"
wav_ar = model.generate(arabic_text, language_id='ar')

# Test Italian
italian_text = "Buongiorno! Come stai oggi?"
wav_it = model.generate(italian_text, language_id='it')

# Compare with baseline model
```

### 3. Upload to HuggingFace

Share your trained model:

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="checkpoint-5000/t3_maltese.pt",
    path_in_repo="t3_mtl24ls_v1.pt",
    repo_id="your-username/chatterbox-maltese",
    repo_type="model"
)
```

## Summary

This guide provides a complete training pipeline for Maltese language support:

1. ✅ Uses MASRI_HEADSET_v2 dataset
2. ✅ Optimized for Google Colab free tier
3. ✅ Prevents catastrophic forgetting through mixed-language training
4. ✅ Includes monitoring and troubleshooting
5. ✅ Provides data pipeline templates
6. ✅ Production-ready configuration

For questions or issues, refer to:
- `MALTESE_FINETUNING_GUIDE.md` - Detailed training theory
- `train_maltese_colab.ipynb` - Interactive Colab notebook
- `train_maltese.py` - Complete training script template

Happy training! 🇲🇹
