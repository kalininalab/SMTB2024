from typing import Literal

from datasets import Dataset, DatasetDict
from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordLevel, WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordLevelTrainer, WordPieceTrainer
from transformers import PreTrainedTokenizerFast

# ! Current Data for token is AI


def train_tokenizer(
    dataset: Dataset | DatasetDict,
    tokenization_type: Literal["bpe", "wordpiece", "unigram", "wordlevel"] = "bpe",
    output_file_directory: str = "data/tokenizer.json",
    vocab_size: int = 5000,
):
    if tokenization_type == "bpe":
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=vocab_size)
    elif tokenization_type == "wordpiece":
        tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
        trainer = WordPieceTrainer(
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=vocab_size
        )
    elif tokenization_type == "unigram":
        tokenizer = Tokenizer(Unigram())
        trainer = UnigramTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=vocab_size)
    elif tokenization_type == "wordlevel":
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=vocab_size
        )

    tokenizer.pre_tokenizer = Whitespace()  # Necessary to avoid \n in tokens

    tokenizer.train_from_iterator(iterator=dataset["train"]["Sequence"], trainer=trainer)

    tokenizer.save(output_file_directory)  # Saves to token.json

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=output_file_directory)

    return tokenizer
