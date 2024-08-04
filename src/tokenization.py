import os
from typing import Literal

import transformers
from datasets import Dataset, DatasetDict
from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordLevel, WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordLevelTrainer, WordPieceTrainer
from transformers import PreTrainedTokenizerFast

try:
    from src.charactertokenizer import CharacterTokenizer
except ModuleNotFoundError:
    from charactertokenizer import CharacterTokenizer

# ! Current Data for token is AI


def train_tokenizer(
    dataset: Dataset | DatasetDict,
    tokenization_type: Literal["bpe", "wordpiece", "unigram", "wordlevel", "char"] = "bpe",
    output_file_directory: str = "data/tokenizer.json",
    vocab_size: int = 5000,
) -> transformers.PreTrainedTokenizerFast | CharacterTokenizer:
    directory = os.path.dirname(output_file_directory)

    if os.path.exists(output_file_directory):
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_file(output_file_directory))
        # tokenizer = PreTrainedTokenizerFast(tokenizer_file=path)
        tokenizer.mask_token = "[MASK]"
        tokenizer.pad_token = "[PAD]"
        return tokenizer
    if not os.path.exists(directory):
        os.makedirs(directory)
    if tokenization_type == "bpe":
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=vocab_size)
        tokenizer.pre_tokenizer = Whitespace()
    elif tokenization_type == "wordpiece":
        tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
        trainer = WordPieceTrainer(
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=vocab_size
        )
        tokenizer.pre_tokenizer = Whitespace()
    elif tokenization_type == "unigram":
        tokenizer = Tokenizer(Unigram())
        trainer = UnigramTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=vocab_size)
        tokenizer.pre_tokenizer = Whitespace()
    elif tokenization_type == "wordlevel":
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=vocab_size
        )
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
        model_max_length = 1024
        tokenizer = CharacterTokenizer(vocab, model_max_length=model_max_length)
        tokenizer.save_pretrained(directory)
        return tokenizer

    tokenizer.train_from_iterator(iterator=dataset["train"]["Sequence"], trainer=trainer)

    tokenizer.save(output_file_directory)

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=output_file_directory)

    # Add special tokens
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
