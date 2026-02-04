# Vocabulary Update Guide for Maltese Language Support

## Overview

To fully enable Maltese language support, the tokenizer vocabulary file needs to be regenerated to include the `[mt]` language token. This guide explains how to update the vocabulary.

## Current Status

- **Current vocabulary file**: `grapheme_mtl_merged_expanded_v1.json` (2454 tokens, 23 languages)
- **Maltese status**: `[mt]` token is currently treated as `[UNK]` (unknown)
- **Target**: Create updated vocabulary with 2455+ tokens including `[mt]`

## Why Vocabulary Update is Needed

The tokenizer vocabulary file (`grapheme_mtl_merged_expanded_v1.json`) is a JSON file that maps text tokens to numeric IDs. Currently:

1. The model supports 23 languages with language tokens: `[ar]`, `[da]`, `[de]`, `[el]`, `[en]`, `[es]`, `[fi]`, `[fr]`, `[he]`, `[hi]`, `[it]`, `[ja]`, `[ko]`, `[ms]`, `[nl]`, `[no]`, `[pl]`, `[pt]`, `[ru]`, `[sv]`, `[sw]`, `[tr]`, `[zh]`
2. When the tokenizer encounters `[mt]`, it treats it as an unknown token
3. For proper Maltese support, `[mt]` needs to be added to the vocabulary

## Steps to Update the Vocabulary

### Option 1: Extend Existing Vocabulary (Recommended)

This approach adds `[mt]` to the existing vocabulary without retraining the entire tokenizer.

#### Step 1: Download Current Vocabulary

```python
from huggingface_hub import hf_hub_download
import json

# Download the current vocabulary file
vocab_file = hf_hub_download(
    repo_id="ResembleAI/chatterbox",
    filename="grapheme_mtl_merged_expanded_v1.json",
    cache_dir="./vocab_cache"
)

# Load vocabulary
with open(vocab_file, 'r', encoding='utf-8') as f:
    vocab_data = json.load(f)

print(f"Current vocabulary type: {vocab_data.get('type', 'unknown')}")
print(f"Vocabulary size: {len(vocab_data.get('model', {}).get('vocab', {}))}")
```

#### Step 2: Add Maltese Token

```python
from tokenizers import Tokenizer

# Load tokenizer
tokenizer = Tokenizer.from_file(vocab_file)

# Get current vocabulary
vocab = tokenizer.get_vocab()
current_size = len(vocab)
print(f"Current vocab size: {current_size}")

# Add [mt] token if not present
if "[mt]" not in vocab:
    # The tokenizer library doesn't directly support adding tokens
    # We need to modify the JSON structure
    
    # Load raw JSON
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab_json = json.load(f)
    
    # Add [mt] token to the vocabulary
    if 'model' in vocab_json and 'vocab' in vocab_json['model']:
        vocab_dict = vocab_json['model']['vocab']
        new_id = max(vocab_dict.values()) + 1
        vocab_dict['[mt]'] = new_id
        print(f"Added [mt] with ID: {new_id}")
        
        # Save updated vocabulary
        output_file = "grapheme_mtl_merged_expanded_v2.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(vocab_json, f, ensure_ascii=False, indent=2)
        
        print(f"Updated vocabulary saved to: {output_file}")
        print(f"New vocabulary size: {len(vocab_dict)}")
    else:
        print("Error: Unexpected vocabulary structure")
else:
    print("[mt] token already exists in vocabulary")
```

#### Step 3: Verify Updated Vocabulary

```python
# Load and test the updated tokenizer
from tokenizers import Tokenizer

updated_tokenizer = Tokenizer.from_file("grapheme_mtl_merged_expanded_v2.json")
updated_vocab = updated_tokenizer.get_vocab()

# Verify [mt] token exists
if "[mt]" in updated_vocab:
    print(f"✓ [mt] token added successfully with ID: {updated_vocab['[mt]']}")
else:
    print("✗ [mt] token not found")

# Test encoding
test_text = "[mt]Bonġu! Kif int illum?"
encoded = updated_tokenizer.encode(test_text)
print(f"Test encoding: {encoded.tokens[:5]}")
print(f"Token IDs: {encoded.ids[:5]}")
```

#### Step 4: Update Model Configuration

After creating the updated vocabulary file:

1. **Upload to HuggingFace Hub** (if you have access):
   ```python
   from huggingface_hub import HfApi
   
   api = HfApi()
   api.upload_file(
       path_or_fileobj="grapheme_mtl_merged_expanded_v2.json",
       path_in_repo="grapheme_mtl_merged_expanded_v2.json",
       repo_id="ResembleAI/chatterbox",
       repo_type="model",
       token="your_hf_token"
   )
   ```

2. **Or use locally**: Update `mtl_tts.py` to use the local file:
   ```python
   # In from_local method
   tokenizer = MTLTokenizer(
       str(ckpt_dir / "grapheme_mtl_merged_expanded_v2.json")
   )
   ```

#### Step 5: Resize Model Embeddings

After updating the vocabulary, resize the model embeddings:

```python
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Load model with updated tokenizer
model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")

# Resize embeddings to match new vocabulary size (2455)
new_vocab_size = 2455  # 2454 + 1 for [mt]
model.t3.resize_text_token_embeddings(new_vocab_size)

print(f"Model embeddings resized to {new_vocab_size}")

# Save updated model checkpoint
# (This step saves the resized model for future use)
import torch
torch.save(model.t3.state_dict(), "t3_mtl24ls_v1.safetensors")
```

### Option 2: Retrain Tokenizer (More Complex)

This approach involves retraining the tokenizer from scratch with Maltese data included.

#### Requirements

1. Text data for all 24 languages (including Maltese)
2. Balanced corpus (similar amount of text per language)
3. Tokenizers training library

#### Process

```python
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from tokenizers.normalizers import NFKD, Lowercase, Sequence

# Define special tokens
special_tokens = [
    "[START]", "[STOP]", "[UNK]", "[SPACE]", "[PAD]", 
    "[SEP]", "[CLS]", "[MASK]",
    # Language tokens (24 languages)
    "[ar]", "[da]", "[de]", "[el]", "[en]", "[es]", 
    "[fi]", "[fr]", "[he]", "[hi]", "[it]", "[ja]", 
    "[ko]", "[ms]", "[mt]",  # Added Maltese
    "[nl]", "[no]", "[pl]", "[pt]", "[ru]", "[sv]", 
    "[sw]", "[tr]", "[zh]"
]

# Create tokenizer with BPE model
tokenizer = Tokenizer(models.BPE())

# Set normalizers (lowercase + NFKD)
tokenizer.normalizer = Sequence([NFKD(), Lowercase()])

# Set pre-tokenizer
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

# Train tokenizer
trainer = trainers.BpeTrainer(
    vocab_size=2500,  # Target vocab size
    special_tokens=special_tokens,
    min_frequency=2
)

# Training files (one per language)
training_files = [
    "data/ar_corpus.txt",
    "data/da_corpus.txt",
    # ... all 24 languages including mt_corpus.txt
]

tokenizer.train(files=training_files, trainer=trainer)

# Save new tokenizer
tokenizer.save("grapheme_mtl24_merged_v1.json")
```

**Note**: This approach requires:
- Maltese text corpus (10-100MB recommended)
- Text corpora for all other languages
- May result in different token IDs (requires model retraining)

## Recommended Approach

**For minimal disruption, use Option 1 (Extend Existing Vocabulary)**:

1. ✅ Only adds one new token (`[mt]`)
2. ✅ Preserves all existing token IDs
3. ✅ Works with existing trained model (just resize embeddings)
4. ✅ Can be done without large text corpora
5. ✅ Fast and straightforward

**Use Option 2 only if**:
- You want to optimize vocabulary for Maltese
- You have access to large Maltese corpus
- You plan to retrain the entire model anyway

## After Vocabulary Update

Once the vocabulary is updated:

1. **Test the tokenizer**:
   ```python
   from chatterbox.models.tokenizers import MTLTokenizer
   
   tokenizer = MTLTokenizer("grapheme_mtl_merged_expanded_v2.json")
   
   # Test Maltese encoding
   maltese_text = "Bonġu! Kif int illum?"
   tokens = tokenizer.encode(maltese_text, language_id="mt")
   print(f"Encoded tokens: {tokens}")
   
   # Verify [mt] token is not [UNK]
   decoded = tokenizer.decode(tokens)
   print(f"Decoded: {decoded}")
   ```

2. **Fine-tune the model** following `MALTESE_FINETUNING_GUIDE.md`

3. **Update documentation** to reflect that `[mt]` is now properly supported

## Troubleshooting

### Issue: "Token ID out of range"

**Cause**: Model embeddings not resized to match new vocabulary size

**Solution**: 
```python
model.t3.resize_text_token_embeddings(new_vocab_size)
```

### Issue: "[mt] still treated as [UNK]"

**Cause**: Using old vocabulary file

**Solution**: Verify the vocabulary file being loaded:
```python
import json
with open("vocabulary_file.json", 'r') as f:
    vocab = json.load(f)
    if '[mt]' in vocab['model']['vocab']:
        print("✓ [mt] token present")
    else:
        print("✗ [mt] token missing - wrong file?")
```

### Issue: "Model outputs are poor after vocabulary update"

**Cause**: New `[mt]` embedding needs training

**Solution**: Fine-tune the model on Maltese data (see `MALTESE_FINETUNING_GUIDE.md`)

## Summary

To fully enable Maltese support:

1. ✅ **Update vocabulary** (this guide) - Add `[mt]` token
2. ✅ **Resize model embeddings** - Use `resize_text_token_embeddings(2455)`
3. ✅ **Fine-tune model** - Train on Maltese dataset (see `MALTESE_FINETUNING_GUIDE.md`)
4. ✅ **Test and validate** - Verify Maltese synthesis quality

The implementation in this PR provides the infrastructure (steps 2-4). This guide covers step 1 (vocabulary update).
