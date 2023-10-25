from transformers import BartForConditionalGeneration, BartTokenizer, TrainingArguments, Trainer
import datasets

# Load your custom dataset
dataset = datasets.load_dataset('path_to_custom_dataset')

# Initialize the model and tokenizer
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# Tokenize and format the dataset
def tokenize_function(examples):
    return tokenizer(examples["document"], examples["summary"], truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    evaluation_strategy="steps",
    save_steps=10_000,
    output_dir="./custom_bart_model",
    device="cuda"  # or "cuda:0" for a specific GPU
)

# Initialize the Trainer and fine-tune the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    data_collator=data_collator,
)

trainer.train()
