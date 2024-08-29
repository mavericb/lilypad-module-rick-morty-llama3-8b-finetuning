import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

def print_directory_contents(path):
    print(f"Contents of {path}:")
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            print(f"  [DIR] {item}")
            print_directory_contents(item_path)
        else:
            print(f"  [FILE] {item}")

print("main.py started")

# Debug: Print directory contents
print_directory_contents("/")

# Debug: Print directory contents
print_directory_contents("/app")

# Use local paths for model and dataset
model_base_path = "/app/model/models--NousResearch--Meta-Llama-3-8B-Instruct"
dataset_name = "/app/dataset"
new_model = "llama-2-7b-rick-c-137"

# Find the correct model path
model_snapshots = os.path.join(model_base_path, "snapshots")
if os.path.exists(model_snapshots):
    snapshot_dirs = [d for d in os.listdir(model_snapshots) if os.path.isdir(os.path.join(model_snapshots, d))]
    if snapshot_dirs:
        model_name = os.path.join(model_snapshots, sorted(snapshot_dirs)[-1])
    else:
        model_name = model_base_path
else:
    model_name = model_base_path

print(f"Using model path: {model_name}")

# QLoRA parameters
lora_r = 64
lora_alpha = 16
lora_dropout = 0.1

# bitsandbytes parameters
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

# TrainingArguments parameters
output_dir = "/app/results"
num_train_epochs = 1
fp16 = False
bf16 = False
per_device_train_batch_size = 4
per_device_eval_batch_size = 4
gradient_accumulation_steps = 1
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "constant"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 25
logging_steps = 25

# SFT parameters
max_seq_length = None
packing = False
device_map = {"": 0}

# Load dataset from local file
try:
    dataset = load_from_disk(dataset_name)
    print(f"Dataset loaded successfully from {dataset_name}")
    print(f"Dataset info: {dataset}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise

# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Load base model from local files
try:
    print(f"Attempting to load model from {model_name}")
    print_directory_contents(model_name)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        local_files_only=True,
        trust_remote_code=True,
    )
    print(f"Model loaded successfully from {model_name}")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer from local files
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
    print(f"Tokenizer loaded successfully from {model_name}")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    raise

tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
)

# Set supervised fine-tuning parameters
try:
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=packing,
    )
    print("SFTTrainer initialized successfully")
except Exception as e:
    print(f"Error initializing SFTTrainer: {e}")
    raise

# Train model
try:
    print("Starting model training...")
    trainer.train()
    print("Model training completed successfully")
except Exception as e:
    print(f"Error during model training: {e}")
    raise

# Save trained model
try:
    trainer.model.save_pretrained(f"/app/{new_model}")
    print(f"Trained model saved to /app/{new_model}")
except Exception as e:
    print(f"Error saving trained model: {e}")
    raise

# Ignore warnings
logging.set_verbosity(logging.CRITICAL)

# Run text generation pipeline with our new model
try:
    prompt = "Who are you?"
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    print("Text generation result:")
    print(result[0]['generated_text'])
except Exception as e:
    print(f"Error during text generation: {e}")

try:
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=device_map,
        local_files_only=True
    )
    model = PeftModel.from_pretrained(base_model, f"/app/{new_model}")
    model = model.merge_and_unload()
    print("Model merged and unloaded successfully")
except Exception as e:
    print(f"Error merging and unloading model: {e}")
    raise

# Reload tokenizer to save it
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print("Tokenizer reloaded successfully")
except Exception as e:
    print(f"Error reloading tokenizer: {e}")
    raise

# Run text generation pipeline with our new model
try:
    prompt = "What are you doing in your garage?"
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    print("Final text generation result:")
    print(result[0]['generated_text'])
except Exception as e:
    print(f"Error during final text generation: {e}")

print("main.py execution completed")
