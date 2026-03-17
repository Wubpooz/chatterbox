from ..llama_configs import LLAMA_CONFIGS


class T3Config:
    def __init__(self, text_tokens_dict_size=704, is_multilingual=None):
        self.start_text_token = 255
        self.stop_text_token = 0
        self.text_tokens_dict_size = text_tokens_dict_size
        self.max_text_tokens = 2048

        self.start_speech_token = 6561
        self.stop_speech_token = 6562
        self.speech_tokens_dict_size = 8194
        self.max_speech_tokens = 4096

        self.llama_config_name = "Llama_520M"
        self.input_pos_emb = "learned"
        self.speech_cond_prompt_len = 150

        self.encoder_type = "voice_encoder"
        self.speaker_embed_size = 256
        self.use_perceiver_resampler = True
        self.emotion_adv = True

        # Determine whether this configuration should be treated as multilingual.
        # Prefer an explicit flag when provided; otherwise, fall back to a heuristic
        # based on known multilingual vocabulary sizes. This avoids incorrectly
        # classifying very large vocabularies (e.g., GPT-2/Turbo with 50k+ tokens)
        # as multilingual.
        if is_multilingual is not None:
            self._is_multilingual = bool(is_multilingual)
        else:
            # Multilingual models use vocabulary size of 2454+ tokens
            # English-only models use 704 tokens
            # 2454 = 23 languages, 2455+ = 24 languages (with Maltese)
            # Apply an upper bound to avoid misclassifying non-T3 models with
            # very large vocabularies (e.g., GPT-2/Turbo) as multilingual.
            self._is_multilingual = 2454 <= self.text_tokens_dict_size <= 10000

    @property
    def n_channels(self):
        return LLAMA_CONFIGS[self.llama_config_name]["hidden_size"]
    
    @property
    def is_multilingual(self):
        """Return True if this configuration is multilingual."""
        return self._is_multilingual

    @classmethod
    def english_only(cls):
        """Create configuration for English-only TTS model."""
        return cls(text_tokens_dict_size=704, is_multilingual=False)
    
    @classmethod 
    def multilingual(cls):
        """Create configuration for multilingual TTS model (24 languages including Maltese)."""
        # Keeping 2454 for backward compatibility with existing models
        # When retraining with Maltese, this should be increased to 2455+
        return cls(text_tokens_dict_size=2454, is_multilingual=True)
