import os
import evaluate
import pandas as pd
import torch
import csv
import json

from peft import (
    PeftModel
)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

os.environ['HF_TOKEN'] = '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'

device_map = "auto"

base_model =  "mistralai/Mistral-7B-Instruct-v0.2"
checkpoint_path = "/raid/ovod/playground/data/jessan/phytochat/ArCHer/checkpoints/phytochat_mistral_dpo/trainer.pt"
predictions_save_path = "data/predictions/archer_mistral_predictions_dpo.json"

base_model =  "meta-llama/Meta-Llama-3-8B-Instruct"
checkpoint_path = "/raid/ovod/playground/data/jessan/phytochat/ArCHer/checkpoints/phytochat_llama_dpo/trainer.pt"
predictions_save_path = "data/predictions/archer_llama_predictions_dpo.json"

# Load the model for Inference
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Reload model in FP16 and merge it with LoRA weights
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True
)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map=device_map,
    cache_dir='/raid/ovod/playground/data/.cache/huggingface/hub',
)
checkpoint = torch.load(checkpoint_path)

from peft import LoraConfig, TaskType, get_peft_model
lora_config = LoraConfig(
    r=16,
    target_modules=['q_proj', 'v_proj'],
    task_type=TaskType.CAUSAL_LM,
    lora_alpha=64,
    lora_dropout=0.05
)
model = get_peft_model(model, lora_config)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

## Customize how model is called here:
def query_model(question):
    input_ids = tokenizer.encode(f"Question:{question}\nAnswer: ", return_tensors="pt").to('cuda')
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

df = pd.read_parquet("data/dpo/test.parquet")

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
