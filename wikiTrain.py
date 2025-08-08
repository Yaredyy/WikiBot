from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
import torch

# Paths
DATA_PATH = "wiki_clean.txt"
MODEL_SAVE_PATH = "./wiki_gpt2"

# Load tokenizer & model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Create dataset loader
def load_dataset(file_path, tokenizer, block_size=128):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )

dataset = load_dataset(DATA_PATH, tokenizer)

# Split dataset
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training setup
training_args = TrainingArguments(
    output_dir=MODEL_SAVE_PATH,
    overwrite_output_dir=True,
    num_train_epochs=1,       # Adjust for longer training
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    save_steps=5000,
    save_total_limit=2,
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=1000,
    learning_rate=5e-5,
    warmup_steps=500,
    fp16=torch.cuda.is_available(),
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train
trainer.train()
model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)
