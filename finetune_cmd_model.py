import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import os

def load_dataset(train_file: str, val_file: str):
    """Load and prepare the dataset for training."""
    def read_jsonl(file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    train_data = read_jsonl(train_file)
    val_data = read_jsonl(val_file)
    
    return Dataset.from_list(train_data), Dataset.from_list(val_data)

def tokenize_function(examples, tokenizer):
    """Tokenize the examples."""
    # Combine prompt and completion with a separator
    texts = [f"{p} [SEP] {c} [EOS]" for p, c in zip(examples['prompt'], examples['completion'])]
    return tokenizer(texts, truncation=True, max_length=128, padding="max_length")

def train():
    # File paths
    train_file = "train_cmd_data.jsonl"
    val_file = "val_cmd_data.jsonl"
    output_dir = "./cmd_model"
    
    # Check if data files exist
    if not (os.path.exists(train_file) and os.path.exists(val_file)):
        print("Dataset files not found. Please run create_cmd_dataset.py first.")
        return
    
    # Initialize tokenizer and model
    model_name = "distilgpt2"  # Using a smaller model for efficiency
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Add special tokens
    special_tokens = {'sep_token': '[SEP]', 'eos_token': '[EOS]'}
    tokenizer.add_special_tokens(special_tokens)
    
    # Load and prepare datasets
    print("Loading and preparing datasets...")
    train_dataset, val_dataset = load_dataset(train_file, val_file)
    
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
    model.resize_token_embeddings(len(tokenizer))
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
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
        save_total_limit=2,
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
    
    # Save the final model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nModel and tokenizer saved to {output_dir}")
    
    # Move model to device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Test the model
    test_prompts = [
        "list all files in current directory",
        "show network configuration",
        "display running processes",
        "clear the screen",
        "what is my ip address"
    ]
    
    print("\nTesting the model with sample prompts:")
    model.eval()
    with torch.no_grad():
        for prompt in test_prompts:
            input_text = f"{prompt} [SEP]"
            input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
            
            output = model.generate(
                input_ids,
                max_length=50,
                temperature=0.3,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_k=20,
                top_p=0.9,
            )
            
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            command = generated_text.split('[SEP]')[-1].strip()
            print(f"\nPrompt: {prompt}")
            print(f"Generated command: {command}")

if __name__ == "__main__":
    train()
