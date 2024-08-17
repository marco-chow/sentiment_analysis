from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np
import os


ds = load_dataset("Yelp/yelp_review_full")

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

#Define a function to tokenize the text
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

#Tokenize the datasets
tokenized_datasets = ds.map(tokenize_function, batched=True)

#Create a smaller subset of data
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

#Load the model
model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)


training_args = TrainingArguments(output_dir="test_trainer")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
)
