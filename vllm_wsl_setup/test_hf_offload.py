import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

model_id = "google/gemma-4-26B-A4B-it"

print("=========================================================")
print(f" Loading {model_id} via HuggingFace Accelerate")
print("=========================================================")
print("This script uses device_map='auto' to load layers into your")
print("6GB GPU VRAM first, and offloads the remaining inactive")
print("layers into your 16GB of System RAM automatically.")
print("=========================================================\n")

# Ensure the user has authenticated
if not os.environ.get("HF_TOKEN") and not os.path.exists(os.path.expanduser("~/.cache/huggingface/token")):
    print("WARNING: You must set HF_TOKEN or run 'huggingface-cli login' first!")

print("1. Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("2. Loading Model (This will take a while as it downloads ~14GB of weights...)")
# device_map="auto" tells the accelerate library to split the model
# across GPU and CPU RAM automatically.
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

print("\nModel Memory Footprint:")
print(model.hf_device_map)

print("\n✅ Model loaded successfully! Type a prompt to test inference (or 'quit' to exit):")

while True:
    try:
        prompt = input("\nPrompt: ")
        if prompt.lower() in ["quit", "exit", "q"]:
            break
            
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        print("Generating response (this may be slow due to CPU offloading)...")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=150)
            
        print("\nOutput:")
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    except KeyboardInterrupt:
        break
