#!/usr/bin/env python3
"""
Script to update the tokenizer vocabulary to include Maltese language token.

This script downloads the current vocabulary, adds the [mt] token, and saves
the updated vocabulary file.

Usage:
    python update_vocabulary.py [--output OUTPUT_FILE]

Requirements:
    pip install tokenizers huggingface_hub
"""

import argparse
import json
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
except ImportError:
    print("Error: Required packages not installed.")
    print("Please run: pip install tokenizers huggingface_hub")
    sys.exit(1)


def download_current_vocabulary(cache_dir="./vocab_cache"):
    """Download the current vocabulary file from HuggingFace."""
    print("Downloading current vocabulary from HuggingFace...")
    
    try:
        vocab_file = hf_hub_download(
            repo_id="ResembleAI/chatterbox",
            filename="grapheme_mtl_merged_expanded_v1.json",
            cache_dir=cache_dir
        )
        print(f"✓ Downloaded to: {vocab_file}")
        return vocab_file
    except Exception as e:
        print(f"✗ Error downloading vocabulary: {e}")
        print("\nAlternative: If you have the vocabulary file locally, ")
        print("you can specify it with: --input YOUR_VOCAB_FILE.json")
        return None


def add_maltese_token(vocab_file, output_file):
    """Add [mt] token to the vocabulary."""
    print(f"\nLoading vocabulary from: {vocab_file}")
    
    # Load the JSON structure
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab_json = json.load(f)
    
    # Check structure
    if 'model' not in vocab_json or 'vocab' not in vocab_json['model']:
        print("✗ Error: Unexpected vocabulary structure")
        print(f"Found keys: {list(vocab_json.keys())}")
        return False
    
    vocab_dict = vocab_json['model']['vocab']
    current_size = len(vocab_dict)
    print(f"Current vocabulary size: {current_size}")
    
    # Check if [mt] already exists
    if '[mt]' in vocab_dict:
        print("✓ [mt] token already exists in vocabulary")
        print(f"  Token ID: {vocab_dict['[mt]']}")
        return True
    
    # Find the highest token ID
    max_id = max(vocab_dict.values())
    new_id = max_id + 1
    
    # Add [mt] token
    vocab_dict['[mt]'] = new_id
    print(f"✓ Added [mt] token with ID: {new_id}")
    
    # Verify other language tokens exist
    expected_lang_tokens = [
        '[ar]', '[da]', '[de]', '[el]', '[en]', '[es]',
        '[fi]', '[fr]', '[he]', '[hi]', '[it]', '[ja]',
        '[ko]', '[ms]', '[nl]', '[no]', '[pl]', '[pt]',
        '[ru]', '[sv]', '[sw]', '[tr]', '[zh]'
    ]
    
    missing_tokens = [tok for tok in expected_lang_tokens if tok not in vocab_dict]
    if missing_tokens:
        print(f"⚠ Warning: Some expected language tokens are missing: {missing_tokens}")
    else:
        print(f"✓ All 23 original language tokens present")
    
    # Save updated vocabulary
    print(f"\nSaving updated vocabulary to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=2)
    
    new_size = len(vocab_dict)
    print(f"✓ Vocabulary saved successfully")
    print(f"  New vocabulary size: {new_size} (added {new_size - current_size} token)")
    
    return True


def verify_vocabulary(vocab_file):
    """Verify the updated vocabulary file."""
    print(f"\nVerifying vocabulary file: {vocab_file}")
    
    try:
        # Load as tokenizer
        tokenizer = Tokenizer.from_file(vocab_file)
        vocab = tokenizer.get_vocab()
        
        # Check [mt] token
        if '[mt]' in vocab:
            print(f"✓ [mt] token verified with ID: {vocab['[mt]']}")
        else:
            print("✗ [mt] token not found in vocabulary")
            return False
        
        # Test encoding
        test_text = "[mt]Bonġu! Kif int illum?"
        encoded = tokenizer.encode(test_text)
        tokens = encoded.tokens
        ids = encoded.ids
        
        print(f"\nTest encoding of: {test_text}")
        print(f"  First 5 tokens: {tokens[:5]}")
        print(f"  First 5 IDs: {ids[:5]}")
        
        # Verify [mt] is tokenized correctly (not as [UNK])
        if '[mt]' in tokens:
            print("✓ [mt] token correctly recognized (not [UNK])")
            return True
        else:
            print("⚠ Warning: [mt] might have been tokenized differently")
            return True
        
    except Exception as e:
        print(f"✗ Error during verification: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Update tokenizer vocabulary to include Maltese language token"
    )
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help="Input vocabulary file (if not provided, downloads from HuggingFace)"
    )
    parser.add_argument(
        '--output',
        type=str,
        default="grapheme_mtl_merged_expanded_v2.json",
        help="Output vocabulary file (default: grapheme_mtl_merged_expanded_v2.json)"
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default="./vocab_cache",
        help="Cache directory for downloads (default: ./vocab_cache)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Tokenizer Vocabulary Update Script")
    print("Adding Maltese language token [mt]")
    print("=" * 70)
    
    # Get input vocabulary file
    if args.input:
        vocab_file = args.input
        if not Path(vocab_file).exists():
            print(f"✗ Error: Input file not found: {vocab_file}")
            sys.exit(1)
    else:
        vocab_file = download_current_vocabulary(args.cache_dir)
        if not vocab_file:
            sys.exit(1)
    
    # Add Maltese token
    success = add_maltese_token(vocab_file, args.output)
    if not success:
        print("\n✗ Failed to update vocabulary")
        sys.exit(1)
    
    # Verify updated vocabulary
    if verify_vocabulary(args.output):
        print("\n" + "=" * 70)
        print("✓ Vocabulary update completed successfully!")
        print("=" * 70)
        print(f"\nNext steps:")
        print(f"1. Use the updated vocabulary file: {args.output}")
        print(f"2. Resize model embeddings to 2455 tokens:")
        print(f"   model.t3.resize_text_token_embeddings(2455)")
        print(f"3. Fine-tune the model on Maltese data")
        print(f"   (see MALTESE_FINETUNING_GUIDE.md)")
        print("\nFor more details, see VOCABULARY_UPDATE_GUIDE.md")
    else:
        print("\n⚠ Vocabulary updated but verification had issues")
        print("Please check the output file manually")
        sys.exit(1)


if __name__ == '__main__':
    main()
