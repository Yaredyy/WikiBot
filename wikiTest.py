from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

MODEL_PATH = "./wiki_gpt2"

# Load model & tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
model.eval()

if torch.cuda.is_available():
    model = model.to("cuda")

# Chat loop
print("Type 'quit' to exit.")
while True:
    prompt = input("\nYou: ")
    if prompt.lower() in ["quit", "exit"]:
        break

    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs,
        max_length=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Bot: {response}")
