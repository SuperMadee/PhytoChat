# Importing necessary libraries
import gc
import json  # Importing the json library for JSON operations
import re  # Importing the re library for regular expressions
from pprint import pprint  # Importing the pprint function from the pprint module
import pandas as pd  # Importing the pandas library for data manipulation and analysis
import torch  # Importing the torch library for deep learning
from datasets import Dataset, load_dataset, DatasetDict  # Importing Dataset classes and functions from the datasets module
from huggingface_hub import notebook_login  # Importing the notebook_login function from the huggingface_hub module
from peft import LoraConfig, PeftModel  # Importing LoraConfig and PeftModel classes from the peft module
from transformers import (  # Importing classes and functions from the transformers module
    AutoModelForCausalLM,  # Importing the AutoModelForCausalLM class for causal language modeling
    AutoTokenizer,  # Importing the AutoTokenizer class for tokenization
    BitsAndBytesConfig,  # Importing the BitsAndBytesConfig class for bits and bytes configuration
    TrainingArguments,  # Importing the TrainingArguments class for training arguments
    pipeline,  # Importing the pipeline function for text generation
    logging,  # Importing the logging module for logging
)
from trl import SFTTrainer  # Importing the SFTTrainer class from the trl module for supervised fine-tuning

import time  # Importing the time module for time-related operations
import os  # Importing the os module for operating system related operations

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'  # Setting an environment variable to suppress advisory warnings
# Set the Hugging Face API token
# os.environ["HF_TOKEN"] = "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"
os.environ['HF_TOKEN'] = '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'

# Load the training dataset
train_parquet_file = "data/sft/train.parquet"  # File path for the training dataset
validation_parquet_file = "data/sft/validation.parquet"  # File path for the validation dataset

dataset = load_dataset("parquet", data_files={'train': train_parquet_file, 'validation': validation_parquet_file})  # Loading the datasets using the load_dataset function

# The model that you want to train from the Hugging Face hub
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

# Fine-tuned model name
os.makedirs("checkpoints", exist_ok=True)
new_model = "checkpoints/mistral_sft"
# Record the start time
start_time = time.time()

################################################################################
# QLoRA parameters
################################################################################

# LoRA attention dimension
lora_r = 32

# Alpha parameter for LoRA scaling
lora_alpha = 16

# Dropout probability for LoRA layers
lora_dropout = 0.1

################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
#use_4bit = False # don't quantize for now
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "bfloat16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = True

################################################################################
# TrainingArguments parameters
################################################################################

# Output directory where the model predictions and checkpoints will be stored
output_dir = "./mistral_sft_results"

# Number of training epochs
num_train_epochs = 4

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = True

# Batch size per GPU for training
per_device_train_batch_size = 20

# Batch size per GPU for evaluation
per_device_eval_batch_size = 4

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 4

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.2

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-5

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# Optimizer to use
optim = "paged_adamw_8bit"

# Learning rate schedule (constant a bit better than cosine)
lr_scheduler_type = "constant"

# Number of training steps (overrides num_train_epochs)
max_steps = -1

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 50

# Log every X updates steps
logging_steps = 25

#---added by MSP 05/30---
eval_strategy = 'steps'
eval_steps = 100

################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
max_seq_length = 128

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False

# Load the entire model on the GPU 0
device_map = {"":0}#{"": device}

# Load the base model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.bfloat16,
    attn_implementation='flash_attention_2',
)

base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

# Load Llama-AI tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Set the evaluation prompt
eval_prompt = """how do you manage citrus canker?"""

# Tokenize the evaluation prompt
model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

# Set the base model to evaluation mode
base_model.eval()

# Generate text using the base model
with torch.no_grad():
    print(tokenizer.decode(base_model.generate(**model_input, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id)[0], skip_special_tokens=True))

torch.cuda.empty_cache()
# --------------------------------------------------------------------------------

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    target_modules=[
        "q_proj",
        # "k_proj",
        "v_proj",
        # "o_proj",
        # "gate_proj",
        # "up_proj",
        # "down_proj",
        # "lm_head",
    ],
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
    report_to="wandb",
    eval_strategy=eval_strategy,
    eval_steps=eval_steps,
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=base_model,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)

# Train the model
trainer.train()

# Save the trained model
trainer.model.save_pretrained(new_model)

del trainer
del base_model
del model_input
del dataset
gc.collect()
torch.cuda.empty_cache()

# Load the base model with specific configurations
print(f"loading base model again")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)

# Merge the base model and the fine-tuned model
print(f"loading peft model")
merged_model= PeftModel.from_pretrained(base_model, new_model)
merged_model= merged_model.merge_and_unload()

# Save the merged model
print(f"saving merged model")
merged_model.save_pretrained("merged_model-8B",safe_serialization=True)
tokenizer.save_pretrained("merged_model-8B")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


del base_model
del merged_model
gc.collect()
torch.cuda.empty_cache()

new_model = 'merged_model-8B'

base_model = AutoModelForCausalLM.from_pretrained(
    new_model,
    # quantization_config=bnb_config,
    device_map={'':0},
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,

)

base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

# Load Llama tokenizer
tokenizer = AutoTokenizer.from_pretrained(new_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# Set the evaluation prompt
eval_prompt = """how do you manage citrus canker?"""

# Tokenize the evaluation prompt
model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

# Set the base model to evaluation mode
base_model.eval()

# Generate text using the base model
with torch.no_grad():
    print(tokenizer.decode(base_model.generate(**model_input, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id, repetition_penalty=1)[0], skip_special_tokens=True))

# # Print training duration
#  # Record the end time
# end_time = time.time()

# # Calculate the elapsed time
# training_time = end_time - start_time

# # Print the training time
# print(f"Training took {training_time} seconds")