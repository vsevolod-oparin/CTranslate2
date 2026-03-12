"""Shared configuration and helpers for Metal e2e tests."""
import os
import sys

# Data directory: set CT2_TEST_DATA env var, or default to ../../../data (sibling of repo root)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("CT2_TEST_DATA", os.path.join(_SCRIPT_DIR, "..", "..", "..", "..", "data"))
DATA_DIR = os.path.normpath(DATA_DIR)


def model_path(name):
    """Return absolute path to a model directory under DATA_DIR."""
    return os.path.join(DATA_DIR, name)


def audio_path(name):
    """Return absolute path to an audio file under DATA_DIR."""
    return os.path.join(DATA_DIR, name)


def load_marian_tokenizer():
    """Load the Helsinki-NLP/opus-mt-en-de MarianTokenizer."""
    from transformers import MarianTokenizer
    return MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")


def tokenize(tokenizer, sentence):
    """Tokenize a sentence into token strings (MarianTokenizer convention)."""
    return tokenizer.convert_ids_to_tokens(tokenizer.encode(sentence))


def decode(tokenizer, token_list):
    """Decode a list of token strings back to text."""
    return tokenizer.decode(tokenizer.convert_tokens_to_ids(token_list))


# --- Whisper language-agnostic helpers ---

# Map CT2 model directory names to HuggingFace processor names
_MODEL_TO_HF_PROCESSOR = {
    "whisper-base":            "openai/whisper-base",
    "whisper-large-v3":        "openai/whisper-large-v3",
    "whisper-large-v3-turbo":  "openai/whisper-large-v3-turbo",
}


def resolve_hf_processor(model_name):
    """Return the HuggingFace processor name for a given CT2 model directory name."""
    if model_name in _MODEL_TO_HF_PROCESSOR:
        return _MODEL_TO_HF_PROCESSOR[model_name]
    return f"openai/{model_name}"


def detect_language_ct2(model, processor, audio_array, sample_rate=16000):
    """Auto-detect audio language using a CTranslate2 Whisper model.

    Works with any Whisper model (base, large-v3, turbo, etc.).
    For English-only models, returns 'en'.

    Args:
        model: ctranslate2.models.Whisper instance
        processor: WhisperProcessor instance
        audio_array: numpy audio array (mono, at sample_rate)
        sample_rate: audio sample rate (default 16000)

    Returns the language code (e.g., 'ru', 'en', 'de').
    """
    import numpy as np
    import ctranslate2

    if not model.is_multilingual:
        return "en"

    chunk_size = 30 * sample_rate
    chunk = audio_array[:chunk_size]
    if len(chunk) < chunk_size:
        chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode="constant")

    inputs = processor(chunk, return_tensors="np", sampling_rate=sample_rate)
    feat = inputs.input_features
    if feat.shape[-1] > 3000:
        feat = feat[:, :, :3000]  # truncate to expected 3000 frames
    features = ctranslate2.StorageView.from_array(feat)
    results = model.detect_language(features)
    best_token, best_prob = results[0][0]
    lang_code = best_token.strip("<|>")
    return lang_code


def detect_language_fw(model, audio_file):
    """Auto-detect audio language using a faster_whisper WhisperModel.

    Args:
        model: faster_whisper.WhisperModel instance
        audio_file: path to audio file

    Returns the language code (e.g., 'ru', 'en', 'de').
    """
    if not model.model.is_multilingual:
        return "en"

    # Use faster_whisper's built-in language detection
    import librosa
    import numpy as np
    import ctranslate2

    audio, _ = librosa.load(audio_file, sr=16000, mono=True)
    chunk = audio[:30 * 16000]
    if len(chunk) < 30 * 16000:
        chunk = np.pad(chunk, (0, 30 * 16000 - len(chunk)), mode="constant")

    features = model.feature_extractor(chunk)
    features = features[:, :3000]  # truncate to expected 3000 frames
    features = np.expand_dims(features, 0)
    features = ctranslate2.StorageView.from_array(features.astype(np.float32))
    results = model.model.detect_language(features)
    best_token, best_prob = results[0][0]
    lang_code = best_token.strip("<|>")
    return lang_code


def get_whisper_prefix_tokens(tokenizer, language, task="transcribe", timestamps=False):
    """Build Whisper prefix tokens from the tokenizer for any model version.

    Returns list of token IDs: [startoftranscript, language, task, notimestamps?]

    All token IDs are derived dynamically from the tokenizer, ensuring
    correctness regardless of vocabulary size or special token offsets.
    """
    sot = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    lang = tokenizer.convert_tokens_to_ids(f"<|{language}|>")
    task_token = tokenizer.convert_tokens_to_ids(f"<|{task}|>")

    prefix = [sot, lang, task_token]
    if not timestamps:
        no_ts = tokenizer.convert_tokens_to_ids("<|notimestamps|>")
        prefix.append(no_ts)

    for i, tok_id in enumerate(prefix):
        if tok_id is None or tok_id < 0:
            raise ValueError(f"Failed to resolve prefix token index {i}: "
                             f"got {tok_id} for prefix {prefix}")
    return prefix


def get_whisper_prefix_strings(language, task="transcribe", timestamps=False):
    """Build Whisper prefix as token strings (for model.generate with string prompts).

    Returns list like ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"]
    """
    prefix = ["<|startoftranscript|>", f"<|{language}|>", f"<|{task}|>"]
    if not timestamps:
        prefix.append("<|notimestamps|>")
    return prefix
