import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import os

def load_dataset(train_path, val_path):
    """Load training and validation datasets."""
    def read_jsonl(file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    train_data = read_jsonl(train_path)
    val_data = read_jsonl(val_path)
    
    # Convert to datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    return train_dataset, val_dataset

def tokenize_function(examples, tokenizer):
    """Tokenize the examples."""
    # Combine prompt and completion with a separator
    texts = [f"{p} [SEP] {c} [EOS]" for p, c in zip(examples['prompt'], examples['completion'])]
    return tokenizer(texts, truncation=True, max_length=128, padding="max_length")

def train():
    # Initialize tokenizer and model
    model_name = "distilgpt2"  # Small and efficient model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Add special tokens if they don't exist
    special_tokens = {'sep_token': '[SEP]', 'eos_token': '[EOS]'}
    tokenizer.add_special_tokens(special_tokens)
    
    # Load and prepare datasets
    train_dataset, val_dataset = load_dataset('train_data.jsonl', 'val_data.jsonl')
    
    # Tokenize datasets
    train_tokenized = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=['prompt', 'completion']
    )
    val_tokenized = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=['prompt', 'completion']
    )
    
    # Initialize model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))  # Resize for new tokens
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./command_model",
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal language modeling
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save the model and tokenizer
    model.save_pretrained("./command_model")
    tokenizer.save_pretrained("./command_model")
    print("Model and tokenizer saved to ./command_model")

if __name__ == "__main__":
    train()
