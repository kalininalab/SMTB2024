import os
from typing import Literal, Union

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

# TODO: Replace example, generated by AI token txt files with actual ones
# ! Current Data for token is AI


def train_tokenizer(
    dataset: Union[Dataset, DatasetDict],
    tokenization_type: Literal["bpe", "wordpiece", "unigram", "wordlevel", "char"] = "bpe",
    output_file_directory: str = "data/tokenizer.json",
    vocab_size: int = 5000,
) -> transformers.PreTrainedTokenizerFast:
    """
    Train a tokenizer on a given dataset.

    Args:
        dataset (Union[Dataset, DatasetDict]): The dataset to train the tokenizer on.
        tokenization_type (Literal["bpe", "wordpiece", "unigram", "wordlevel", "char"]): The type of tokenizer to train. Default is "bpe".
        output_file_directory (str): The directory to save the tokenizer file. Default is "data/tokenizer.json".
        vocab_size (int): The size of the vocabulary. Default is 5000.

    Returns:
        transformers.PreTrainedTokenizerFast: The trained tokenizer.
    """
    directory = os.path.dirname(output_file_directory)
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
        output_file_directory = f"{directory.rstrip('/')}/tokenizer_config.json"
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=output_file_directory)
        return tokenizer

    tokenizer.train_from_iterator(iterator=dataset["train"]["Sequence"], trainer=trainer)

    tokenizer.save(output_file_directory)

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=output_file_directory)

    return tokenizer
