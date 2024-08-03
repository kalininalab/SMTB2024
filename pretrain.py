from argparse import ArgumentParser

from datasets import DatasetDict, load_dataset
from transformers import (
    DataCollatorForLanguageModeling,
    EsmConfig,
    EsmForMaskedLM,
    Trainer,
    TrainingArguments,
)

from src.tokenization import train_tokenizer

parser = ArgumentParser()
parser.add_argument("--dataset", type=str, default="khairi/uniprot-swissprot")
parser.add_argument("--vocab_size", type=int, default=5000)
args = parser.parse_args()

dataset = load_dataset(args.dataset)
# Replace with your file path


def rename_column_if_needed(example):
    if "text" not in example.keys():
        example["text"] = example.pop("Sequence")
    # also remove any other columns that are not needed
    return example


dataset = dataset.map(rename_column_if_needed)

if isinstance(dataset, DatasetDict):
    for split in dataset.keys():
        dataset[split] = dataset[split].map(rename_column_if_needed)


# tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_file("data/bpe.json"))
tokenizer = train_tokenizer(dataset["train"]["text"], vocab_size=args.vocab_size)
# tokenizer.mask_token = "[MASK]"
# tokenizer.pad_token = "[PAD]"

tokenized_datasets = dataset.map(
    lambda x: tokenizer(x["text"], max_length=255, truncation=True, padding="max_length"),
    batched=False,
    num_proc=8,
    remove_columns=["text", "EntryID", "__index_level_0__"],
)


config = EsmConfig(
    vocab_size=args.vocab_size,  # Same as vocab size you used for the tokenizer
    hidden_size=320,
    num_hidden_layers=6,
    num_attention_heads=4,
    max_position_embeddings=256,
    pad_token_id=0,
    mask_token_id=1,
)

model = EsmForMaskedLM(config=config)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir="./esm-from-scratch",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
    # no_cuda=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
)

trainer.train()
