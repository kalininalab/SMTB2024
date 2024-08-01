from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling, EsmConfig, EsmForMaskedLM, Trainer, TrainingArguments

from src.tokenization import train_tokenizer

dataset = load_dataset("khairi/uniprot-swissprot", split="train[:10%]+validation[:10%]+test[:10%]")
# Replace with your file path

tokenizer = train_tokenizer(dataset["train"]["text"], vocab_size=50)
tokenizer.mask_token = "[MASK]"
tokenizer.pad_token = "[PAD]"

tokenized_datasets = dataset.map(lambda x: tokenizer(x["Sequence"]), batched=True, num_proc=4, remove_columns=["text"])


config = EsmConfig(
    vocab_size=5000,  # Same as vocab size you used for the tokenizer
    hidden_size=128,
    num_hidden_layers=6,
    num_attention_heads=4,
    max_position_embeddings=512,
    pad_token_id=0,
    mask_token_id=1,
)

model = EsmForMaskedLM(config=config)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15, pad_to_multiple_of=512)

training_args = TrainingArguments(
    output_dir="./esm-from-scratch",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
    no_cuda=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
)

trainer.train()
