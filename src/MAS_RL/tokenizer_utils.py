"""Word-level tokenizer utilities."""

from __future__ import annotations

import os

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer


PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"


def train_word_tokenizer(texts: list[str], min_frequency: int = 1) -> Tokenizer:
    tokenizer = Tokenizer(WordLevel(unk_token=UNK_TOKEN))
    tokenizer.normalizer = Lowercase()
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(
        min_frequency=min_frequency,
        show_progress=False,
        special_tokens=[PAD_TOKEN, UNK_TOKEN],
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.enable_padding(pad_id=0, pad_token=PAD_TOKEN)
    return tokenizer


def load_tokenizer(path: str) -> Tokenizer:
    tokenizer = Tokenizer.from_file(path)
    tokenizer.enable_padding(pad_id=0, pad_token=PAD_TOKEN)
    return tokenizer


def save_tokenizer(tokenizer: Tokenizer, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tokenizer.save(path)


def encode_texts(tokenizer: Tokenizer, texts: list[str], max_length: int) -> tuple[list[list[int]], list[list[int]]]:
    tokenizer.enable_truncation(max_length=max_length)
    tokenizer.enable_padding(length=max_length, pad_id=0, pad_token=PAD_TOKEN)
    encoded = tokenizer.encode_batch(texts)
    input_ids = [item.ids for item in encoded]
    attention_mask = [item.attention_mask for item in encoded]
    return input_ids, attention_mask
