import argparse
import os

import pytorch_lightning as pl
import torch
from datasets import DatasetDict, load_dataset
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, EsmConfig, EsmForMaskedLM

from src.tokenization import train_tokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Add this for better error messages

parser = argparse.ArgumentParser(description="Pretrain a model")
parser.add_argument("--data", type=str, default="khairi/uniprot-swissprot", help="Name of data to be trained on")
parser.add_argument("--tokenizer", default="bpe", type=str, choices=["char", "bpe"], help="Tokenizer to use")
parser.add_argument("--vocab_size", default=5000, type=int, help="Vocabulary size")
parser.add_argument("--n_layers", type=int, default=12, help="Number of layers in the model")
parser.add_argument("--n_dims", type=int, default=128, help="Dimensions of the model")
parser.add_argument("--n_heads", type=int, default=4, help="Number of heads in the model")
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

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

## Load the dataset ##
print("Loading dataset...")
dataset = load_dataset(config.data)
print(f"Dataset loaded: {dataset}")


def rename_column_if_needed(example):
    if "text" not in example.keys():
        example["text"] = example.pop("Sequence")
    return example


dataset = dataset.map(rename_column_if_needed)

if isinstance(dataset, DatasetDict):
    for split in dataset.keys():
        dataset[split] = dataset[split].map(rename_column_if_needed)

### Train the tokenizer & Load the pretrained tokenizer ###
print("Training tokenizer...")
tokenizer = train_tokenizer(
    dataset=dataset["train"]["text"],
    tokenization_type=config.tokenizer,
    vocab_size=config.vocab_size,
    output_file_directory=config.token_output_file,
)
print(f"Tokenizer trained and saved at: {config.token_output_file}")

tokenized_datasets = dataset.map(
    lambda x: tokenizer(x["text"], max_length=255, truncation=True, padding="max_length"),
    batched=True,
    num_proc=8,
    remove_columns=["text", "EntryID", "__index_level_0__"],
)

print("Tokenized datasets created.")
print(f"Example tokenized input: {tokenized_datasets['train'][0]}")

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

model = EsmForMaskedLM(config=esm_config)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)


# Define a PyTorch Lightning DataModule
class MyDataModule(pl.LightningDataModule):
    def __init__(self, tokenized_datasets, batch_size):
        super().__init__()
        self.tokenized_datasets = tokenized_datasets
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = self.tokenized_datasets["train"]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=24,  # Adjust as needed
            pin_memory=True,
        )


# Define a PyTorch Lightning Module
class MyLightningModule(pl.LightningModule):
    def __init__(self, model, data_collator, learning_rate=1e-4):
        super().__init__()
        self.model = model
        self.data_collator = data_collator
        self.learning_rate = learning_rate

    def training_step(self, batch, batch_idx):
        # del batch["token_type_ids"]
        # print(type(batch["input_ids"]), len(batch["input_ids"]))
        # print(batch["input_ids"])
        tmp = self.data_collator(batch)
        t_batch = {
            "input_ids": torch.stack(batch["input_ids"]).long(),
            "attention_mask": torch.stack(batch["attention_mask"]).float(),
        }
        outputs = self.model(**t_batch)
        print(outputs)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


data_module = MyDataModule(tokenized_datasets, config.batch_size)
lightning_model = MyLightningModule(model, data_collator)

# Setup PyTorch Lightning Trainer
trainer = pl.Trainer(
    max_epochs=config.epochs,
    devices=[1],
    default_root_dir=config.output_dir,
    callbacks=[ModelCheckpoint(monitor="train_loss")],
)

print("Starting training...")
trainer.fit(lightning_model, datamodule=data_module)
print("Training completed.")
