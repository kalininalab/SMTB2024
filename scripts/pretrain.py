import argparse
import os

from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling, EsmConfig, EsmForMaskedLM, Trainer, TrainingArguments

from smtb.tokenization import train_tokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "true"

parser = argparse.ArgumentParser(description="Pretrain a model")
parser.add_argument("--tokenizer", default="bpe", type=str, choices=["char", "bpe"], help="Tokenizer to use")
parser.add_argument("--vocab_size", default=5000, type=int, help="Vocabulary size")
parser.add_argument("--n_layers", type=int, default=6, help="Number of layers in the model")
parser.add_argument("--n_dims", type=int, default=320, help="Dimensions of the model")
parser.add_argument("--n_heads", type=int, default=16, help="Number of heads in the model")
parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to train")
parser.add_argument("--batch_size", type=int, default=200, help="Batch size")
parser.add_argument("--output_dir", type=str, default="output", help="Output directory for model and checkpoints")
parser.add_argument(
    "--token_output_file",
    type=str,
    default="output/tokenizer/tokenizer.json",
    help="Where the tokenizer json file will be saved",
)
config = parser.parse_args()

## Load the dataset ##
dataset = load_dataset("text", data_files={"train": "/scratch/pretrain/swissprot.txt"})

### Train the tokenizer & Load the pretrained tokenizer ###
# You can choose the tokenizer type, default is bpe
tokenizer = train_tokenizer(
    dataset=dataset,
    tokenization_type=config.tokenizer,
    vocab_size=config.vocab_size,
    output_file_directory=config.token_output_file,
)

# Add padding token to the tokenizer if not already present
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})


### Tokenize the dataset
def tokenize_function(examples: dict) -> dict:
    max_length = 250  # Define the maximum length for the sequences
    tokens = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)
    return tokens


dataset = dataset.map(tokenize_function, batched=False, num_proc=16, remove_columns=["text"])
print(dataset)


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
).to("cuda")

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir=config.output_dir,
    num_train_epochs=config.epochs,
    per_device_train_batch_size=config.batch_size,
    bf16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
)

trainer.train()
