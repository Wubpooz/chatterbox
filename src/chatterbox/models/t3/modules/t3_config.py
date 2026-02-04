from ..llama_configs import LLAMA_CONFIGS


class T3Config:
    def __init__(self, text_tokens_dict_size=704):
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

    @property
    def n_channels(self):
        return LLAMA_CONFIGS[self.llama_config_name]["hidden_size"]
    
    @property
    def is_multilingual(self):
        # Updated to include Maltese (24 languages total)
        # Original: 2454 tokens for 23 languages
        # New: 2455+ tokens for 24 languages (added [mt] token)
        return self.text_tokens_dict_size >= 2454

    @classmethod
    def english_only(cls):
        """Create configuration for English-only TTS model."""
        return cls(text_tokens_dict_size=704)
    
    @classmethod 
    def multilingual(cls):
        """Create configuration for multilingual TTS model (24 languages including Maltese)."""
        # Keeping 2454 for backward compatibility with existing models
        # When retraining with Maltese, this should be increased to 2455+
        return cls(text_tokens_dict_size=2454)
