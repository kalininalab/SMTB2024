from pathlib import Path
from typing import Iterable, Literal

from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordLevel, WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordLevelTrainer, WordPieceTrainer
from transformers import PreTrainedTokenizerFast


def train_tokenizer(
    data: Iterable[str],
    output_dir: Path,
    tokenization_type: Literal["bpe", "wordpiece", "unigram", "wordlevel", "char"] = "bpe",
    vocab_size: int = 5000,
) -> PreTrainedTokenizerFast:
    special_tokens = ["[PAD]", "[MASK]", "[UNK]", "[CLS]", "[SEP]"]
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    if tokenization_type == "bpe":
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        trainer = BpeTrainer(special_tokens=special_tokens, vocab_size=vocab_size)
        tokenizer.pre_tokenizer = Whitespace()
    elif tokenization_type == "wordpiece":
        tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
        trainer = WordPieceTrainer(special_tokens=special_tokens, vocab_size=vocab_size)
        tokenizer.pre_tokenizer = Whitespace()
    elif tokenization_type == "unigram":
        tokenizer = Tokenizer(Unigram())
        trainer = UnigramTrainer(special_tokens=special_tokens, vocab_size=vocab_size)
        tokenizer.pre_tokenizer = Whitespace()
    elif tokenization_type == "wordlevel":
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        trainer = WordLevelTrainer(special_tokens=special_tokens, vocab_size=vocab_size)
        tokenizer.pre_tokenizer = Whitespace()
    elif tokenization_type == "char":
        vocab = [
            "L",
            "A",
            "G",
            "V",
            "S",
            "E",
            "R",
            "T",
            "I",
            "D",
            "P",
            "K",
            "Q",
            "N",
            "F",
            "Y",
            "M",
            "H",
            "W",
            "C",
            "X",
            "B",
            "U",
            "Z",
            "O",
            ".",
            "-",
        ]
        tokenizer = PreTrainedTokenizerFast()
        tokenizer.add_tokens(vocab)
    if tokenization_type != "char":
        tokenizer.train_from_iterator(iterator=data[:1000], trainer=trainer)
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    tokenizer.mask_token = "[MASK]"
    tokenizer.pad_token = "[PAD]"
    return tokenizer
