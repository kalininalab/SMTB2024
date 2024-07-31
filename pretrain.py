import argparse

from transformers import PreTrainedTokenizerFast

import pandas as pd

parser = argparse.ArgumentParser(description="Pretrain a model")
parser.add_argument("--data", type=str, help=".txt file containing the data")
parser.add_argument("--tokenizer", type=str, choices=["char", "bpe"], help="Tokenizer to use")
parser.add_argument("--vocab_size", type=int, help="Vocabulary size")
parser.add_argument("--n_layers", type=int, default=12, help="Number of layers in the model")
parser.add_argument("--n_dims", type=int, default=480, help="Dimensions of the model")
parser.add_argument("--n_heads", type=int, default=16, help="Number of heads in the model")
parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
config = parser.parse_args()


## Load the dataset ##

dataset = ...


### Train the tokenizer


### Load the pretrained tokenizer


tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=...,
    mask_token="[MASK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    unk_token="[UNK]",
)


### Tokenize the dataset


### Init the config and the model

config = ...
model = ...
data_collator = ...

### Setup trainer ###

training_args = ...
trainer = ...
trainer.train()
