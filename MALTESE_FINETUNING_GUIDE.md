# Fine-Tuning Guide for Maltese Language Support

## Overview

This guide provides comprehensive instructions for fine-tuning the Chatterbox multilingual TTS model to add proper Maltese language support. It addresses the vocabulary expansion method, training strategies, and explains the decision between full fine-tuning vs. parameter-efficient methods like LoRA.

## Table of Contents
1. [Understanding the Resize Method](#understanding-the-resize-method)
2. [Full Fine-Tuning vs. LoRA](#full-fine-tuning-vs-lora)
3. [Recommended Training Strategy](#recommended-training-strategy)
4. [Step-by-Step Training Process](#step-by-step-training-process)
5. [Alternative: LoRA Fine-Tuning](#alternative-lora-fine-tuning)
6. [Monitoring and Evaluation](#monitoring-and-evaluation)

---

## Understanding the Resize Method

### Is the Resize Method Sound?

**Yes, the `resize_text_token_embeddings()` method is a standard and well-established approach** used in NLP for vocabulary expansion. Here's why:

#### What It Does
```python
# For NEW tokens (like [mt]):
new_embedding = mean(existing_embeddings) + std(existing_embeddings) * random_noise
```

#### Why This Works
1. **Preserves Existing Knowledge**: All existing token embeddings (ar, it, en, etc.) remain unchanged
2. **Smart Initialization**: New tokens start from a reasonable position in the embedding space (not random)
3. **Distribution Matching**: New embeddings have similar statistics to existing ones, making training more stable
4. **Standard Practice**: This is the same method used by Hugging Face's `resize_token_embeddings()` and similar libraries

#### Similar Methods in Literature
- **BERT vocabulary expansion** (Devlin et al., 2019)
- **XLM-R language adaptation** (Conneau et al., 2020)
- **mT5 tokenizer extension** (Xue et al., 2021)

All use mean/std initialization or similar techniques for new tokens.

---

## Full Fine-Tuning vs. LoRA

### When to Use Full Fine-Tuning (Recommended for Maltese)

**For adding a new language, full fine-tuning is recommended** over LoRA. Here's why:

#### Advantages of Full Fine-Tuning for New Languages

1. **New Token Embeddings Need Training**
   - The `[mt]` token embedding is randomly initialized (with smart mean/std)
   - LoRA only updates adapter weights, **not embedding layers by default**
   - New language tokens need direct embedding updates to learn proper representations

2. **Cross-Lingual Knowledge Transfer**
   - Full fine-tuning allows the model to learn connections between:
     - Maltese `[mt]` ↔ Arabic `[ar]` (Semitic roots)
     - Maltese `[mt]` ↔ Italian `[it]` (Romance vocabulary)
   - These connections happen in the embedding and attention layers
   - LoRA primarily affects transformer attention layers, missing embedding-level transfer

3. **Small Dataset Scenario**
   - For Maltese, you likely have a **small to medium dataset** (< 100 hours)
   - With proper regularization, full fine-tuning works well on small datasets
   - LoRA shines when you have **tiny datasets** (< 10 hours) or computational constraints

4. **Text Embeddings Are Small**
   - Text vocabulary: ~2454 tokens, embedding dim: 512-768
   - Total parameters: ~1-2M (only 0.3% of the 500M model)
   - Training cost is negligible compared to full model

#### When LoRA Makes Sense

LoRA is better for:
- **Domain adaptation** (same language, different style)
- **Speaker adaptation** (voice cloning with minimal data)
- **Extremely limited compute** (training on CPU or small GPU)
- **Many parallel adaptations** (need 10+ language variants simultaneously)

For **adding a new language with vocabulary expansion**, full fine-tuning is the right choice.

---

## Recommended Training Strategy

### Architecture Components to Train

```
┌─────────────────────────────────────────────────────────┐
│                   Chatterbox Model                       │
├─────────────────────────────────────────────────────────┤
│ 1. Text Tokenizer & Embeddings      [TRAIN: YES]       │
│    - text_emb (2454 → 2455+ tokens)                     │
│    - text_head (output projection)                       │
│    Status: NEW [mt] token needs training                │
│                                                           │
│ 2. T3 Transformer (Text→Speech)     [TRAIN: YES]       │
│    - LLaMA/GPT2 backbone                                │
│    - Position embeddings                                 │
│    Status: Fine-tune to learn Maltese patterns          │
│                                                           │
│ 3. Voice Encoder                    [TRAIN: NO]         │
│    - Speaker embedding extraction                        │
│    Status: Language-independent, no training needed     │
│                                                           │
│ 4. S3 Speech Tokenizer              [TRAIN: NO]         │
│    - Speech token quantization                           │
│    Status: Language-independent, no training needed     │
│                                                           │
│ 5. S3Gen Decoder                    [TRAIN: NO]         │
│    - Speech token → Mel spectrogram                      │
│    Status: Language-independent, no training needed     │
└─────────────────────────────────────────────────────────┘
```

### Training Configuration

#### What to Train
```python
# Freeze speech encoder/decoder (as you noted, they don't need updates)
for param in model.ve.parameters():
    param.requires_grad = False

for param in model.s3gen.parameters():
    param.requires_grad = False

# Train text embeddings and T3 model
for param in model.t3.text_emb.parameters():
    param.requires_grad = True

for param in model.t3.text_head.parameters():
    param.requires_grad = True

for param in model.t3.tfmr.parameters():
    param.requires_grad = True
```

#### Recommended Hyperparameters

```python
training_config = {
    # Optimizer
    "optimizer": "AdamW",
    "learning_rate": 1e-5,  # Lower LR to avoid forgetting
    "weight_decay": 0.01,
    "betas": (0.9, 0.999),
    
    # Learning rate schedule
    "scheduler": "cosine",
    "warmup_steps": 500,
    "max_steps": 10000,  # Adjust based on dataset size
    
    # Batch size
    "batch_size": 16,  # Adjust for your GPU
    "gradient_accumulation_steps": 4,  # Effective batch size: 64
    
    # Regularization (prevent forgetting)
    "dropout": 0.1,
    "max_grad_norm": 1.0,
    
    # Multi-language training
    "language_sampling_alpha": 0.7,  # Up-sample Maltese
    "include_original_languages": True,  # Mix with ar, it, en
    "maltese_ratio": 0.4,  # 40% Maltese, 60% other languages
}
```

---

## Step-by-Step Training Process

### Phase 1: Prepare Data (Maltese + Related Languages)

#### 1.1 Maltese Dataset
```python
# Expected format:
maltese_data = [
    {
        "audio": "path/to/audio1.wav",  # 10-second clips
        "text": "Bonġu! Kif int illum?",
        "language": "mt",
        "speaker_id": "speaker_001"
    },
    # ... more samples
]

# Target: 10-50 hours of Maltese speech
# Quality > Quantity: Clean, natural speech preferred
```

#### 1.2 Include Related Languages (Prevent Forgetting)
```python
# Include some Arabic and Italian data in training
related_languages_data = [
    {"audio": "...", "text": "...", "language": "ar"},  # 10% of dataset
    {"audio": "...", "text": "...", "language": "it"},  # 10% of dataset
    {"audio": "...", "text": "...", "language": "en"},  # 5% of dataset
]

# Total mix: 40% mt, 35% ar, 20% it, 5% en
```

### Phase 2: Initialize Model with Resized Embeddings

```python
import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Load pre-trained model
model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")

# Resize embeddings for new vocabulary
# This should be done BEFORE training if tokenizer vocabulary was regenerated
# If [mt] token is in vocabulary, resize to match:
new_vocab_size = 2455  # 2454 + 1 for [mt]
model.t3.resize_text_token_embeddings(new_vocab_size)

# Freeze speech modules
model.ve.requires_grad_(False)
model.s3gen.requires_grad_(False)

# Verify trainable parameters
trainable_params = sum(p.numel() for p in model.t3.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
```

### Phase 3: Training Loop (Simplified)

```python
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Optimizer
optimizer = AdamW(
    [p for p in model.t3.parameters() if p.requires_grad],
    lr=1e-5,
    weight_decay=0.01
)

# Scheduler
scheduler = CosineAnnealingLR(optimizer, T_max=10000)

# Training loop
model.t3.train()
for step, batch in enumerate(train_loader):
    # Forward pass
    # batch contains: text_tokens, speech_tokens, speaker_emb, language_id
    loss_text, loss_speech = model.t3.loss(
        t3_cond=batch['t3_cond'],
        text_tokens=batch['text_tokens'],
        text_token_lens=batch['text_token_lens'],
        speech_tokens=batch['speech_tokens'],
        speech_token_lens=batch['speech_token_lens']
    )
    
    # Combined loss
    loss = loss_text + loss_speech
    
    # Backward pass
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.t3.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    
    # Logging
    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")
    
    # Validation every 1000 steps
    if step % 1000 == 0:
        validate_on_maltese(model, val_loader)
```

### Phase 4: Evaluation and Checkpointing

```python
def validate_on_maltese(model, val_loader):
    """Test on Maltese, Arabic, and Italian to check forgetting"""
    model.t3.eval()
    
    test_samples = {
        "mt": "Bonġu! Kif int illum?",
        "ar": "مرحبا، كيف حالك اليوم؟",
        "it": "Buongiorno! Come stai oggi?",
        "en": "Hello! How are you today?"
    }
    
    with torch.no_grad():
        for lang, text in test_samples.items():
            wav = model.generate(text, language_id=lang)
            # Save audio for manual inspection
            # Compute metrics: MOS, speaker similarity, intelligibility
    
    model.t3.train()
```

---

## Alternative: LoRA Fine-Tuning

### When to Use LoRA (Optional)

If you decide to use LoRA despite the recommendations above, here's how:

#### Step 1: Add PEFT Dependency

```bash
pip install peft
```

#### Step 2: Apply LoRA to T3 Model

```python
from peft import LoraConfig, get_peft_model

# Configure LoRA
lora_config = LoraConfig(
    r=16,  # LoRA rank (smaller = fewer parameters)
    lora_alpha=32,  # LoRA scaling
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Apply to attention
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA to T3 transformer
model.t3.tfmr = get_peft_model(model.t3.tfmr, lora_config)

# Still need to train embeddings separately!
model.t3.text_emb.requires_grad_(True)
model.t3.text_head.requires_grad_(True)

# Print trainable parameters
model.t3.tfmr.print_trainable_parameters()
```

#### LoRA Training Notes

1. **Embeddings must still be trained directly** (LoRA doesn't affect them)
2. Use **higher learning rate** for LoRA adapters (1e-4)
3. Use **lower learning rate** for embeddings (1e-5)
4. Training is faster but may have slightly lower quality for new language

---

## Monitoring and Evaluation

### Key Metrics to Track

#### 1. Training Loss
- Text token prediction loss (should decrease)
- Speech token prediction loss (should decrease)
- Combined loss trend

#### 2. Language-Specific Evaluation

```python
# Test on held-out Maltese data
maltese_eval = {
    "phoneme_error_rate": ...,  # Should be < 15%
    "word_error_rate": ...,      # Should be < 20%
    "speaker_similarity": ...,   # Should be > 0.75
    "naturalness_mos": ...,      # Should be > 3.5/5.0
}

# Test on Arabic/Italian to check forgetting
arabic_eval = {...}   # Should not degrade > 5%
italian_eval = {...}  # Should not degrade > 5%
```

#### 3. Code-Switching Performance

```python
# Test mixed-language sentences
code_switch_test = [
    ("Il-università tagħna", "mt", "has a great program", "en"),
    ("La mia famiglia vive a", "it", "Malta", "mt"),
]
```

### Signs of Good Training

✅ **Good Signs:**
- Maltese loss decreases steadily
- Arabic/Italian performance stays within 95% of baseline
- Generated Maltese audio sounds natural
- Code-switching works smoothly

❌ **Warning Signs:**
- Arabic/Italian performance drops > 10% → reduce Maltese ratio
- Maltese loss plateaus early → increase learning rate or data
- Generated audio has artifacts → check data quality

---

## Conclusion

### Summary of Recommendations

1. **Resize Method**: ✅ Sound and standard practice
2. **Training Approach**: Full fine-tuning (not LoRA) for new language
3. **What to Train**: Text embeddings + T3 transformer
4. **What to Freeze**: Voice encoder + Speech tokenizer + S3Gen decoder
5. **Dataset**: 40% Maltese + 60% related languages (ar, it, en)
6. **Learning Rate**: 1e-5 (conservative to prevent forgetting)
7. **Duration**: 10K steps for 10-50 hours of Maltese data

### Final Notes

The `resize_text_token_embeddings()` method provides a **solid foundation** for adding Maltese. The key to success is:

1. **Good data quality** (clean Maltese audio + transcripts)
2. **Mixed training** (include Arabic and Italian to maintain knowledge)
3. **Conservative learning rate** (prevent catastrophic forgetting)
4. **Regular validation** (check all languages, not just Maltese)

**LoRA is not necessary for this task** and may actually hurt performance for vocabulary expansion. Save LoRA for domain/speaker adaptation tasks.

For questions or issues during training, the community can help optimize hyperparameters based on your specific dataset size and quality.
