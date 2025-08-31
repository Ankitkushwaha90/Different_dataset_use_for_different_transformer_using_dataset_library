# Practical Hugging Face Transformer Examples 🚀

In this guide, we’ll explore **practical use cases** of Hugging Face `datasets` + `transformers` with the right `AutoModelFor...` classes.

---

## 🔹 1. Sequence Classification (Sentiment Analysis – IMDb)

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# Load IMDb dataset
dataset = load_dataset("imdb")

# Load tokenizer and model
checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# Tokenize function
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Take small sample for demo
small_train = tokenized_dataset["train"].shuffle(seed=42).select(range(1000))
small_test = tokenized_dataset["test"].shuffle(seed=42).select(range(200))

# Training setup
args = TrainingArguments("test-sentiment", evaluation_strategy="epoch", per_device_train_batch_size=8)
trainer = Trainer(model=model, args=args, train_dataset=small_train, eval_dataset=small_test)

trainer.train()
```
✅ Task → Sentiment Analysis (classification).

## 🔹 2. Causal Language Modeling (Text Generation – WikiText-2)
```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

# Load WikiText dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

checkpoint = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)

# Tokenize
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length")

tokenized_dataset = dataset.map(tokenize_function, batched=True)

small_train = tokenized_dataset["train"].shuffle(seed=42).select(range(1000))

args = TrainingArguments("test-causal-lm", evaluation_strategy="epoch", per_device_train_batch_size=4)
trainer = Trainer(model=model, args=args, train_dataset=small_train)

trainer.train()
```
✅ Task → Next-word prediction / text generation.

## 🔹 3. Question Answering (Extractive QA – SQuAD)
```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer

# Load SQuAD dataset
dataset = load_dataset("squad")

checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)

# Tokenize context + question
def preprocess(example):
    return tokenizer(example["question"], example["context"], truncation=True, padding="max_length")

tokenized_dataset = dataset.map(preprocess, batched=True)

small_train = tokenized_dataset["train"].shuffle(seed=42).select(range(500))
small_val = tokenized_dataset["validation"].shuffle(seed=42).select(range(100))

args = TrainingArguments("test-qa", evaluation_strategy="epoch", per_device_train_batch_size=8)
trainer = Trainer(model=model, args=args, train_dataset=small_train, eval_dataset=small_val)

trainer.train()
```
✅ Task → Extract answer span from context.

## 🔹 4. Seq2Seq Language Modeling (Summarization – CNN/DailyMail)
```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer

# Load CNN/DailyMail summarization dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")

checkpoint = "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

# Preprocess
def preprocess(example):
    model_inputs = tokenizer(example["article"], max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(example["highlights"], max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess, batched=True)

small_train = tokenized_dataset["train"].shuffle(seed=42).select(range(1000))
small_val = tokenized_dataset["validation"].shuffle(seed=42).select(range(200))

args = TrainingArguments("test-summarization", evaluation_strategy="epoch", per_device_train_batch_size=2)
trainer = Trainer(model=model, args=args, train_dataset=small_train, eval_dataset=small_val)

trainer.train()
```
✅ Task → Summarization (Seq2Seq).
## 🔹 Summary Table
| Transformer Class                    | Task                        | Example Dataset |
| ------------------------------------ | --------------------------- | --------------- |
| `AutoModelForSequenceClassification` | Sentiment / Classification  | IMDb            |
| `AutoModelForCausalLM`               | Text Generation             | WikiText-2      |
| `AutoModelForQuestionAnswering`      | Extractive QA               | SQuAD           |
| `AutoModelForSeq2SeqLM`              | Summarization / Translation | CNN/DailyMail   |
## ⚡ Each code above does:

- Load dataset via Hugging Face datasets library

- Preprocess with AutoTokenizer

- Load task-specific AutoModelFor...

- Train using Trainer
