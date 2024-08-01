import argparse

from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling, EsmConfig, EsmForMaskedLM, Trainer, TrainingArguments

from src.tokenization import train_tokenizer

parser = argparse.ArgumentParser(description="Pretrain a model")
parser.add_argument("--data", type=str, default="khairi/uniprot-swissprot", help="Name of data to be trained on")
parser.add_argument("--tokenizer", default="bpe", type=str, choices=["char", "bpe"], help="Tokenizer to use")
parser.add_argument("--vocab_size", default=5000, type=int, help="Vocabulary size")
parser.add_argument("--n_layers", type=int, default=12, help="Number of layers in the model")
parser.add_argument("--n_dims", type=int, default=480, help="Dimensions of the model")
parser.add_argument("--n_heads", type=int, default=16, help="Number of heads in the model")
parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument(
    "--output_dir", type=str, default="/scratch/output", help="Output directory for model and checkpoints"
)
parser.add_argument(
    "--token_output_file",
    type=str,
    default="/scratch/output/tokenizer/tokenizer.json",
    help="Where the tokenizer json file will be saved",
)
config = parser.parse_args()


## Load the dataset ##

dataset = load_dataset(config.data)


### Train the tokenizer & Load the pretrained tokenizer ###
# ! Multiple Possible Tokenizers
# TODO: Find the best tokenizer

# You can choose the tokenizer type, default is bpe
tokenizer = train_tokenizer(
    dataset=dataset,
    tokenization_type=config.tokenizer,
    vocab_size=config.vocab_size,
    output_file_directory=config.token_output_file,
)


### Tokenize the dataset
def tokenize_function(examples: dict) -> dict:
    max_length = 1024  # Define the maximum length for the sequences
    tokens = tokenizer.encode_batch(examples["Sequence"])
    truncated_tokens = [token_ids[:max_length] for token_ids in tokens]
    return {"ids": truncated_tokens}


# Apply the tokenization function to the dataset
dataset = dataset.map(tokenize_function, batched=True)

### Setup Model ###

esm_config = EsmConfig(
    vocab_size=config.vocab_size,
    num_hidden_layers=config.n_layers,
    hidden_size=config.n_dims,
    num_attention_heads=config.n_heads,
)

model = EsmForMaskedLM(
    config=esm_config,
)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

### Setup trainer ###

training_args = TrainingArguments(
    output_dir=config.output_dir,
    num_train_epochs=config.epochs,
    per_device_train_batch_size=config.batch_size,
)

#! Using swissprot (Small DB)
# The dataset is composed of multiple files: test.parquet, train.parquet, and valid.parquet each composed of different sequences. The function/library I'm using (datasets.load_dataset) parses it into different attributes in a class. You can get them by fetching train, test, or valid.

# In addition, each file is composed of different columns, the one we need being sequence which is why we narrow down our search to it.
# About the structure: https://chatgpt.com/share/488dafd8-f8a9-4280-8a07-d7661273af8e
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    tokenizer=tokenizer,
    train_dataset=dataset["train"]["Sequence"],
    eval_dataset=dataset["validation"]["Sequence"],
)
trainer.train()
