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
