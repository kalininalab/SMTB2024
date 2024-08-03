import argparse
import os
from pathlib import Path

from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling, EsmConfig, EsmForMaskedLM, Trainer, TrainingArguments

from src.tokenization import train_tokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "true"

parser = argparse.ArgumentParser(description="Pretrain a model")
parser.add_argument("--data", type=str, default="khairi/uniprot-swissprot", help="Name of data to be trained on")
parser.add_argument("--tokenizer", default="bpe", type=str, help="Tokenizer to use")
parser.add_argument("--vocab_size", default=5000, type=int, help="Vocabulary size")
parser.add_argument("--n_layers", type=int, default=12, help="Number of layers in the model")
parser.add_argument("--n_dims", type=int, default=480, help="Dimensions of the model")
parser.add_argument("--n_heads", type=int, default=16, help="Number of heads in the model")
parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
parser.add_argument("--output_dir", type=Path, default="data", help="Output directory for model and checkpoints")
config = parser.parse_args()

## Load the dataset ##
dataset = load_dataset(config.data)

### Train the tokenizer & Load the pretrained tokenizer ###
# You can choose the tokenizer type, default is bpe
tokenizer = train_tokenizer(
    data=dataset["train"]["Sequence"],
    tokenization_type=config.tokenizer,
    vocab_size=config.vocab_size,
    output_dir=config.output_dir,
)

# Add padding token to the tokenizer if not already present
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})


### Tokenize the dataset
def tokenize_function(examples: dict) -> dict:
    max_length = 255  # Define the maximum length for the sequences
    tokens = tokenizer(examples["Sequence"], padding="max_length", truncation=True, max_length=max_length)
    return tokens


dataset = dataset.map(tokenize_function, batched=True, remove_columns=["Sequence"])

### Setup Model ###
esm_config = EsmConfig(
    vocab_size=config.vocab_size,
    num_hidden_layers=config.n_layers,
    hidden_size=config.n_dims,
    num_attention_heads=config.n_heads,
    pad_token_id=tokenizer.pad_token_id,
    max_position_embeddings=256,
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
    use_cpu=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)

trainer.train()
