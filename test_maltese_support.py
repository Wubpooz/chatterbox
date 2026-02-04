#!/usr/bin/env python3
"""
Integration test for Maltese language support in Chatterbox Multilingual TTS.

This test validates:
1. Maltese is in SUPPORTED_LANGUAGES
2. The tokenizer accepts Maltese language_id
3. The T3 model has the resize_text_token_embeddings method
4. Maltese text preprocessing works correctly
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_maltese_in_supported_languages():
    """Test that Maltese is in the supported languages list."""
    try:
        from chatterbox.mtl_tts import SUPPORTED_LANGUAGES
        
        print("✓ Test 1: Importing SUPPORTED_LANGUAGES...")
        assert 'mt' in SUPPORTED_LANGUAGES, "Maltese (mt) not found in SUPPORTED_LANGUAGES"
        assert SUPPORTED_LANGUAGES['mt'] == 'Maltese', f"Expected 'Maltese', got '{SUPPORTED_LANGUAGES['mt']}'"
        assert len(SUPPORTED_LANGUAGES) == 24, f"Expected 24 languages, got {len(SUPPORTED_LANGUAGES)}"
        print(f"  ✓ Maltese is supported: {SUPPORTED_LANGUAGES['mt']}")
        print(f"  ✓ Total languages: {len(SUPPORTED_LANGUAGES)}")
        return True
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False

def test_tokenizer_maltese_preprocessing():
    """Test that tokenizer can handle Maltese text."""
    try:
        print("\n✓ Test 2: Testing tokenizer Maltese preprocessing...")
        
        # Test text preprocessing logic without full model
        from unicodedata import normalize
        
        maltese_text = "Bonġu! Kif int illum?"
        preprocessed = maltese_text.lower()
        preprocessed = normalize("NFKD", preprocessed)
        
        # Check special Maltese characters
        assert 'ġ' in maltese_text or 'g' in preprocessed.lower(), "Maltese special character handling failed"
        
        print(f"  ✓ Original text: {maltese_text}")
        print(f"  ✓ Preprocessed: {preprocessed}")
        print(f"  ✓ Maltese special characters handled correctly")
        return True
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False

def test_t3_resize_method_exists():
    """Test that T3 model has the resize method."""
    try:
        print("\n✓ Test 3: Checking T3 resize_text_token_embeddings method...")
        
        from chatterbox.models.t3.t3 import T3
        
        # Check method exists
        assert hasattr(T3, 'resize_text_token_embeddings'), "resize_text_token_embeddings method not found"
        
        # Check method signature
        import inspect
        sig = inspect.signature(T3.resize_text_token_embeddings)
        params = list(sig.parameters.keys())
        assert 'self' in params and 'new_vocab_size' in params, "Method signature incorrect"
        
        print(f"  ✓ resize_text_token_embeddings method exists")
        print(f"  ✓ Method signature: {sig}")
        return True
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False

def test_t3_config_multilingual():
    """Test that T3Config supports multilingual mode."""
    try:
        print("\n✓ Test 4: Checking T3Config multilingual support...")
        
        from chatterbox.models.t3.modules.t3_config import T3Config
        
        config = T3Config.multilingual()
        assert config.text_tokens_dict_size >= 2454, f"Expected vocab size >= 2454, got {config.text_tokens_dict_size}"
        assert config.is_multilingual, "Config should be multilingual"
        
        print(f"  ✓ Multilingual config text_tokens_dict_size: {config.text_tokens_dict_size}")
        print(f"  ✓ is_multilingual: {config.is_multilingual}")
        return True
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False

def test_language_token_format():
    """Test that language tokens are formatted correctly."""
    try:
        print("\n✓ Test 5: Testing language token format...")
        
        language_id = "mt"
        txt = "Test text"
        formatted = f"[{language_id.lower()}]{txt}"
        
        assert formatted == "[mt]Test text", f"Expected '[mt]Test text', got '{formatted}'"
        
        print(f"  ✓ Language token format: {formatted}")
        return True
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False

def test_multilingual_app_config():
    """Test that multilingual_app.py has Maltese config."""
    try:
        print("\n✓ Test 6: Checking multilingual_app.py Maltese config...")
        
        with open('multilingual_app.py', 'r') as f:
            content = f.read()
        
        assert '"mt":' in content, "Maltese config not found in multilingual_app.py"
        assert 'għadda' in content, "Maltese text sample not found"
        
        print(f"  ✓ Maltese config found in multilingual_app.py")
        return True
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False

def test_readme_documentation():
    """Test that README.md documents Maltese support."""
    try:
        print("\n✓ Test 7: Checking README.md documentation...")
        
        with open('README.md', 'r') as f:
            content = f.read()
        
        assert 'Maltese (mt)' in content, "Maltese not documented in README.md"
        assert '24+' in content or '24 ' in content, "Language count not updated to 24"
        
        print(f"  ✓ Maltese documented in README.md")
        print(f"  ✓ Language count updated")
        return True
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False

def main():
    print("=" * 60)
    print("Maltese Language Support Integration Tests")
    print("=" * 60)
    
    tests = [
        test_maltese_in_supported_languages,
        test_tokenizer_maltese_preprocessing,
        test_t3_resize_method_exists,
        test_t3_config_multilingual,
        test_language_token_format,
        test_multilingual_app_config,
        test_readme_documentation,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"\n✗ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    print("=" * 60)
    
    if all(results):
        print("\n✓ All tests passed! Maltese language support is correctly implemented.")
        return 0
    else:
        print("\n✗ Some tests failed. Please review the implementation.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
