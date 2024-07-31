import argparse

from datasets import load_dataset

from src.tokenizers import train_tokenizer
from transformers import PreTrainedTokenizerFast

parser = argparse.ArgumentParser(description="Pretrain a model")
parser.add_argument("--data", type=str, default="khairi/uniprot-swissprot", help="Name of data to be trained on")
parser.add_argument("--tokenizer", default="bpe", type=str, choices=["char", "bpe"], help="Tokenizer to use")
parser.add_argument("--vocab_size", default=5000, type=int, help="Vocabulary size")
parser.add_argument("--n_layers", type=int, default=12, help="Number of layers in the model")
parser.add_argument("--n_dims", type=int, default=480, help="Dimensions of the model")
parser.add_argument("--n_heads", type=int, default=16, help="Number of heads in the model")
parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
config = parser.parse_args()

# Load the dataset
dataset = load_dataset(config.data)

# Train the tokenizer
tokenizer = train_tokenizer(dataset=dataset, tokenization_type=config.tokenizer, vocab_size=config.vocab_size)

def tokenize_function(examples):
    tokens = tokenizer.encode_batch(examples["text"])
    return {"ids": [t.ids for t in tokens]}

# Apply the tokenization function to the dataset
dataset = dataset.map(tokenize_function, batched=True)

# Load the pretrained tokenizer
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=...,
    mask_token="[MASK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    unk_token="[UNK]",
)

# Initialize the config and model
config = ...
model = ...
data_collator = ...

# Setup trainer
training_args = ...
trainer = ...
trainer.train()
