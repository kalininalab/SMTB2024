from pathlib import Path
from typing import Iterable, Literal

import transformers
from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordPiece, Model
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordPieceTrainer, Trainer
from transformers import PreTrainedTokenizerFast

TOKENIZATION_TYPES = Literal["bpe", "wordpiece", "unigram", "char"]


def _get_tokenizer(
    model: Model,
    trainer: Trainer,
    vocab_size: int,
    model_kwargs: dict | None = None,
    trainer_kwargs: dict | None = None,
) -> tuple[Tokenizer, Trainer]:
    """
    Helper function to get tokenizer and trainer objects.

    Args:
        model (Model): model object to be initialized
        trainer (Trainer): trainer object to be initialized
        vocab_size (int): How many tokens to learn
        model_kwargs (dict | None): Arguments to be passed to the model
        trainer_kwargs (dict | None): Arguments to be passed to the trainer

    Return:
        Initialized tokenizer and trainer objects
    """
    if model_kwargs is None:
        model_kwargs = dict(unk_token="[UNK]")
    if trainer_kwargs is None:
        trainer_kwargs = dict(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=vocab_size)
    tokenizer = Tokenizer(model(**model_kwargs))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = trainer(**trainer_kwargs)
    return tokenizer, trainer


def train_tokenizer(
    dataset: Iterable[str],
    tokenization_type: TOKENIZATION_TYPES = "bpe",
    output_dir: str | Path = "data/tokenization",
    vocab_size: int = 5000,
) -> transformers.PreTrainedTokenizerFast:
    """
    Train a tokenizer on a given dataset.

    Args:
        dataset (Iterable[str]): Dataset to train the tokenizer on
        tokenization_type (TOKENIZER_TYPES): Type of tokenization to use
        output_dir (str | Path): Directory to save the tokenizer
        vocab_size (int): How many tokens to learn

    Return:
         Trained tokenizer
    """
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    match tokenization_type:
        case "bpe":
            tokenizer, trainer = _get_tokenizer(BPE, BpeTrainer, vocab_size)
        case "wordpiece":
            tokenizer, trainer = _get_tokenizer(WordPiece, WordPieceTrainer, vocab_size)
        case "unigram":
            tokenizer, trainer = _get_tokenizer(Unigram, UnigramTrainer, vocab_size, model_kwargs={})
        case "char":
            # Char-level tokenization, 5 special tokens and 20 aminoacids
            tokenizer = Tokenizer(
                BPE(
                    vocab={
                        "[PAD]": 0,
                        "[MASK]": 1,
                        "[CLS]": 2,
                        "[SEP]": 3,
                        "[UNK]": 4,
                        "A": 5,
                        "C": 6,
                        "D": 7,
                        "E": 8,
                        "F": 9,
                        "G": 10,
                        "H": 11,
                        "I": 12,
                        "K": 13,
                        "L": 14,
                        "M": 15,
                        "N": 16,
                        "P": 17,
                        "Q": 18,
                        "R": 19,
                        "S": 20,
                        "T": 21,
                        "V": 22,
                        "W": 23,
                        "Y": 24,
                    },
                    merges=[],
                    unk_token="[UNK]",
                )
            )
    # Train the tokenizer if it's not a char-level tokenizer that should only use aminoacids as tokens
    if tokenization_type != "char":
        tokenizer.train_from_iterator(iterator=dataset, trainer=trainer)

    # save the tokenizer ...
    tokenizer_file = str(output_dir / f"{tokenization_type}.json")
    tokenizer.save(tokenizer_file)

    # ... and load it into a PreTrainedTokenizerFast object (from transformers package)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
    tokenizer.add_special_tokens(
        {
            "pad_token": "[PAD]",
            "mask_token": "[MASK]",
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
            "unk_token": "[UNK]",
        }
    )
    return tokenizer
