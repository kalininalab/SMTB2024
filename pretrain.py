import argparse
import os

from datasets import DatasetDict, load_dataset
from transformers import DataCollatorForLanguageModeling, EsmConfig, EsmForMaskedLM, Trainer, TrainingArguments

from src.tokenization import train_tokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "true"

parser = argparse.ArgumentParser(description="Pretrain a model")
parser.add_argument("--data", type=str, default="khairi/uniprot-swissprot", help="Name of data to be trained on")
parser.add_argument("--tokenizer", default="bpe", type=str, choices=["char", "bpe"], help="Tokenizer to use")
parser.add_argument("--vocab_size", default=5000, type=int, help="Vocabulary size")
parser.add_argument("--n_layers", type=int, default=12, help="Number of layers in the model")
parser.add_argument("--n_dims", type=int, default=480, help="Dimensions of the model")
parser.add_argument("--n_heads", type=int, default=16, help="Number of heads in the model")
parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--output_dir", type=str, default="data/output", help="Output directory for model and checkpoints")
parser.add_argument(
    "--token_output_file",
    type=str,
    default="data/output/tokenizer/tokenizer.json",
    help="Where the tokenizer json file will be saved",
)
config = parser.parse_args()

## Load the dataset ##
dataset = load_dataset(config.data)


def rename_column_if_needed(example):
    if "text" not in example.keys():
        example["text"] = example.pop("Sequence")
    # also remove any other columns that are not needed
    return example


dataset = dataset.map(rename_column_if_needed)

if isinstance(dataset, DatasetDict):
    for split in dataset.keys():
        dataset[split] = dataset[split].map(rename_column_if_needed)


### Train the tokenizer & Load the pretrained tokenizer ###
# You can choose the tokenizer type, default is bpe
tokenizer = train_tokenizer(
    dataset=dataset["train"]["text"],
    tokenization_type=config.tokenizer,
    vocab_size=config.vocab_size,
    output_file_directory=config.token_output_file,
)

tokenized_datasets = dataset.map(
    lambda x: tokenizer(x["text"], max_length=255, truncation=True, padding="max_length"),
    batched=False,
    num_proc=8,
    remove_columns=["text", "EntryID", "__index_level_0__"],
)

# Remove the length column
dataset = dataset.remove_columns(["EntryID", "__index_level_0__"])

### Setup Model ###
esm_config = EsmConfig(
    vocab_size=config.vocab_size,
    num_hidden_layers=config.n_layers,
    hidden_size=config.n_dims,
    num_attention_heads=config.n_heads,
    max_position_embeddings=256,
    pad_token_id=tokenizer.pad_token_id,
    mask_token_id=tokenizer.mask_token_id,
)

model = EsmForMaskedLM(
    config=esm_config,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

### Setup trainer ###
training_args = TrainingArguments(
    output_dir=config.output_dir,
    overwrite_output_dir=True,
    num_train_epochs=config.epochs,
    per_device_train_batch_size=config.batch_size,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    # eval_dataset=dataset["validation"],
)

trainer.train()
