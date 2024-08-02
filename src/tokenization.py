import os
from typing import Literal

from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordLevel, WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordLevelTrainer, WordPieceTrainer
from transformers import PreTrainedTokenizerFast


def train_tokenizer(
    dataset,
    tokenization_type: Literal["bpe", "wordpiece", "unigram", "wordlevel"] = "bpe",
    out_dir: str = "data",
    vocab_size: int = 5000,
):
    special_tokens = ["[PAD]", "[MASK]", "[UNK]", "[CLS]", "[SEP]"]
    path = os.path.join(out_dir, f"{tokenization_type}.json")
    if os.path.exists(path):
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=path)
        tokenizer.mask_token = "[MASK]"
        tokenizer.pad_token = "[PAD]"
        return tokenizer
    if tokenization_type == "bpe":
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        trainer = BpeTrainer(
            special_tokens=special_tokens,
            vocab_size=vocab_size,
        )
    elif tokenization_type == "wordpiece":
        tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
        trainer = WordPieceTrainer(special_tokens=special_tokens, vocab_size=vocab_size)
    elif tokenization_type == "unigram":
        tokenizer = Tokenizer(Unigram())
        trainer = UnigramTrainer(special_tokens=special_tokens, vocab_size=vocab_size)
    elif tokenization_type == "wordlevel":
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        trainer = WordLevelTrainer(special_tokens=special_tokens, vocab_size=vocab_size)
    tokenizer.pre_tokenizer = Whitespace()  # Necessary to avoid \n in tokens
    tokenizer.train_from_iterator(iterator=dataset, trainer=trainer)
    tokenizer.save(path)
    tokenizer = PreTrainedTokenizerFast(path)
    tokenizer.mask_token = "[MASK]"
    tokenizer.pad_token = "[PAD]"
    return tokenizer
