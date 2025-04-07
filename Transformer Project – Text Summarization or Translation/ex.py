from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer

# Load dataset for summarization
dataset =  load_dataset("cnn_dailymail", "3.0.0")
print(dataset["train"][0])

# For translation
# dataset = load_dataset("wmt14", "en-fr")

tokenizer =  AutoTokenizer.from_pretrained("t5-small")

# Tokenizer for summarization
def tokenize_function(examples):
    inputs = ["summarize: "+ doc for doc in examples["articles"]]
    model_inputs =  tokenizer(inputs, max_length=512, truncation=True)
    
    #Tokenizer targets
    with tokenizer.as_target_tokenizer():
        labels =  tokenizer(examples["highlights"], max_length=150, truncation=True)
    
    model_inputs["labels"] = labels['input_ids']
    return model_inputs

tokenized_datasets= dataset.map(tokenize_function, batched=True)

model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

training_args =  TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_eval_batch_size=8,
    per_gpu_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=.01,
    save_total_limit=2,
    predict_with_generate=True
)    

trainer =  Trainer(
    model=model,
    args=training_args, 
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets['validation'],
    processing_class=tokenizer
)

trainer.train()

sample_text =  "The Transfomer model has revolutionized NLP by enabling parallel processing of sequences."
inputs = tokenizer("summarize: " + sample_text, return_tensors="pt", max_length=512, truncation=True)
outputs =  model.generate(inputs["input_ids"], max_length=150, num_beams=4, early_stopping=True)

print("generated Summary: ", tokenizer.decode(outputs[0], skip_special_token=True))

# metric = load_metric("rogue")
# prediction = outputs("generated_text")
# references = dataset["validation"]["highlights"]

# results = metric.compute(predictions=predictions, referemces=references)
# print(results)