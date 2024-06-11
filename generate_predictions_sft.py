import os
import evaluate
import pandas as pd
import torch
import csv
import json

from peft import (
    PeftModel
)
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['HF_TOKEN'] = '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'

device_map = "auto"

# mistral
base_model =  "mistralai/Mistral-7B-Instruct-v0.2"
adapter_model = "/raid/ovod/playground/data/jessan/phytochat/checkpoints/mistral_dpo"
predictions_save_path = "data/predictions/dpo_mistral_predictions_sft.json"

# llama
base_model =  "meta-llama/Meta-Llama-3-8B-Instruct"
adapter_model = "/raid/ovod/playground/data/jessan/phytochat/checkpoints/llama_sft"
predictions_save_path = "data/predictions/sft_llama_predictions_sft.json"

# adapter_model = None

# Load the model for Inference
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Reload model in FP16 and merge it with LoRA weights
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float32,
    device_map=device_map,
)
if adapter_model is not None:
    model = PeftModel.from_pretrained(model, adapter_model)
    model = model.merge_and_unload()
    
model.eval()

## Customize how model is called here:
def query_model(question):
    input_ids = tokenizer.encode(f"Answer the question concisely.\n\nQuestion:{question}\nAnswer: ", return_tensors="pt").to('cuda')
    output_ids = model.generate(input_ids, max_new_tokens=64, num_beams=5, no_repeat_ngram_size=2, early_stopping=True, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    try:
        start_index = response.index("Answer: ") + len("Answer: ")
        response = response[start_index:].strip()
    except ValueError:
        response = response.replace(question, "").strip()
    return response

# Query Model For Predictions
from tqdm import tqdm
import os
references = []
predictions = []
questions = []

df = pd.read_parquet("data/sft/test.parquet")

#get total number of rows
total_rows = len(df.axes[0])

data = []
for index, row in tqdm(df.iterrows(), total=total_rows):
    question = row['question']
    questions.append(question)
    gt = str(row['answer']).strip()
    if not gt == '':
        references.append(gt)
        prediction = query_model(question)
        predictions.append(prediction)
        data.append({
            'question': question,
            'reference': gt,
            'prediction': prediction
        })
    with open(predictions_save_path, 'w') as f:
        json.dump(data, f, indent=4)
