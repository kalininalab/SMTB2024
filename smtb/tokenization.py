from pathlib import Path
from typing import Iterable, Literal

import transformers
from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordPieceTrainer
from transformers import PreTrainedTokenizerFast

TOKENIZATION_TYPES = Literal["bpe", "wordpiece", "unigram", "char"]


def _get_tokenizer(
    model,
    trainer,
    vocab_size: int,
    model_kwargs: dict | None = None,
    trainer_kwargs: dict | None = None,
):
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
            tokenizer, trainer = _get_tokenizer(BPE, BpeTrainer, 26)

    tokenizer.train_from_iterator(iterator=dataset, trainer=trainer)
    tokenizer_file = str(output_dir / f"{tokenization_type}.json")
    tokenizer.save(tokenizer_file)
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
