from typing import get_args

import pytest
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerFast

from smtb.tokenization import TOKENIZATION_TYPES, train_tokenizer


def tokenization_types():
    return get_args(TOKENIZATION_TYPES)


@pytest.fixture
def sample_dataset():
    data = {
        "train": {
            "text": [
                "SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRDEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYDSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPVGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGI"
            ],
        }
    }
    return DatasetDict({"train": Dataset.from_dict(data)})


@pytest.mark.parametrize("tokenization_type", tokenization_types())
def test_train_tokenizer(tokenization_type, sample_dataset, tmp_path):
    output_dir = tmp_path / "tokenization"
    tokenizer = train_tokenizer(
        dataset=sample_dataset, tokenization_type=tokenization_type, output_dir=output_dir, vocab_size=50
    )
    assert isinstance(tokenizer, PreTrainedTokenizerFast)
    assert (output_dir / f"{tokenization_type}.json").exists()
    assert tokenizer.pad_token == "[PAD]"
    assert tokenizer.mask_token == "[MASK]"
    assert tokenizer.cls_token == "[CLS]"
    assert tokenizer.sep_token == "[SEP]"
    assert tokenizer.unk_token == "[UNK]"


def test_char_tokenizer(sample_dataset, tmp_path):
    output_dir = tmp_path / "tokenization"
    tokenizer = train_tokenizer(dataset=sample_dataset, tokenization_type="char", output_dir=output_dir)
    for token in tokenizer.vocab.keys():
        if token[0] != "[":
            assert len(token) == 1, f"Token {token} is not a single character"
