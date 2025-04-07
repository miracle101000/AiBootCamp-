from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Load dataset
dataset = load_dataset("imdb")

# Tokenizer
tokenizer  =  AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Prepare data for training
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

train_dataset = tokenized_datasets["train"]
test_dataset =  tokenized_datasets["test"]

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

training_arguments =  TrainingArguments(
    output_dir='./results',
    eval_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=.01,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500
)

trainer = Trainer(
    model=model, 
    args=training_arguments, 
    train_dataset=train_dataset, 
    eval_dataset=test_dataset, 
    processing_class=tokenizer
)

trainer.train()

results =  trainer.evaluate()
print("Evaluation Results:", results)

# Experiment with GPT
from transformers import AutoModelForCasualLM

gpt_model = AutoModelForCasualLM.from_pretrained("gpt2")

input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
output =  gpt_model.generate(input_ids, max_length=50, num_return_sequences=1)

print("generated text: ", tokenizer.decode(output[0], skip_special_token=True))