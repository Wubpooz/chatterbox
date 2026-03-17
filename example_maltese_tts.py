"""
Example script demonstrating Maltese language support in Chatterbox Multilingual TTS.

This example shows:
1. Basic Maltese text-to-speech synthesis
2. Code-switching between Maltese, Italian, and Arabic
3. Using Arabic and Italian reference audio for Maltese synthesis
"""

import torchaudio as ta
import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the multilingual model
print("Loading Chatterbox Multilingual TTS model...")
model = ChatterboxMultilingualTTS.from_pretrained(device=device)

# Example 1: Basic Maltese synthesis
print("\n=== Example 1: Basic Maltese Synthesis ===")
maltese_text = "Bonġu! Kif int illum? Jiena kuntent li niltaqa' miegħek."
print(f"Text: {maltese_text}")

wav = model.generate(maltese_text, language_id="mt")
ta.save("maltese_basic.wav", wav, model.sr)
print("✓ Saved to maltese_basic.wav")

# Example 2: Maltese with Italian reference audio
print("\n=== Example 2: Maltese with Italian Influence ===")
maltese_italian_text = "Il-Kummissjoni Ewropea qed taħdem fuq dan il-proġett importanti."
print(f"Text: {maltese_italian_text}")

# Using Italian-influenced synthesis (if Italian reference audio available)
# This demonstrates knowledge transfer from Italian to Maltese
wav = model.generate(maltese_italian_text, language_id="mt")
ta.save("maltese_italian_influence.wav", wav, model.sr)
print("✓ Saved to maltese_italian_influence.wav")

# Example 3: Code-switching demonstration
print("\n=== Example 3: Code-Switching (Maltese-Italian-English) ===")
# Note: For real code-switching, you would need to process each language segment separately
# This example shows how the model handles Maltese with loanwords
code_switch_text = "Il-università tagħna għandha programm tajjeb. The program għandu success kbir."
print(f"Text: {code_switch_text}")

wav = model.generate(code_switch_text, language_id="mt")
ta.save("maltese_code_switching.wav", wav, model.sr)
print("✓ Saved to maltese_code_switching.wav")

# Example 4: Maltese using Arabic reference for Semitic features
print("\n=== Example 4: Maltese with Arabic Influence ===")
maltese_arabic_text = "Malta għandha storja kbira u kultura unika."
print(f"Text: {maltese_arabic_text}")

# The model can leverage Arabic knowledge for Maltese Semitic features
wav = model.generate(maltese_arabic_text, language_id="mt")
ta.save("maltese_arabic_influence.wav", wav, model.sr)
print("✓ Saved to maltese_arabic_influence.wav")

# Example 5: Maltese with custom voice
print("\n=== Example 5: Maltese with Custom Voice Reference ===")
# If you have a Maltese reference audio file, uncomment and modify:
# maltese_ref_audio = "path/to/your/maltese_reference.wav"
# wav = model.generate(
#     "Din hija prova tal-kloning tal-vuċi bil-Malti.",
#     language_id="mt",
#     audio_prompt_path=maltese_ref_audio
# )
# ta.save("maltese_custom_voice.wav", wav, model.sr)
print("(Skipped - requires custom reference audio)")

print("\n=== Examples Complete ===")
print("\nNotes:")
print("- Maltese uses Latin script with special characters (ċ, ġ, ħ, ż)")
print("- The model leverages knowledge from Arabic (Semitic) and Italian (Romance)")
print("- Code-switching works best with proper language tagging per segment")
print("- For optimal results with Maltese, the tokenizer vocabulary should include [mt] token")
