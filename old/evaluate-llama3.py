import evaluate
import pandas as pd
import torch
import csv

from peft import (
    PeftModel
)
from transformers import AutoModelForCausalLM, AutoTokenizer

device_map = "auto"

# Enter location of base model or adapter_model
base_model =  "meta-llama/Llama-2-13b-chat-hf"
# base_model = "/home/madee/MaroonChat/merged_model-13B
adapter_model = "/home/madee/MaroonChat/merged_model-13B"
# adapter_model = "/home/madee/MaroonChat/dpo_results"
# adapter_model = None

# Load the model for Inference
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Reload model in FP16 and merge it with LoRA weights
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.bfloat16,
    device_map=device_map,
)
if adapter_model is not None:
    model = PeftModel.from_pretrained(model, adapter_model)
    model = model.merge_and_unload()
    
model.eval()

## Customize how model is called here:
def query_model(question):
    input_ids = tokenizer.encode(f"{question}### Response:\n", return_tensors="pt")
    output_ids = model.generate(input_ids.to("cuda"), max_length=200, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    try:
        start_index = response.index("### Response:") + len("### Response:")
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

# read from csv file if existing. Delete this file to re-run predictions
if os.path.exists("maroon_chat_predictions.csv"):
    print("Loading predictions from csv file")
    df = pd.read_csv("maroon_chat_predictions.csv")
    for index, row in tqdm(df.iterrows(), total=len(df.axes[0])):
        questions.append(row['question'])
        references.append(row['reference'])
        predictions.append(row['prediction'])
else:
    df = pd.read_excel("MAROON CHAT Q&A.xlsx", sheet_name="Public")

    #get total number of rows
    total_rows = len(df.axes[0])

    for index, row in tqdm(df.iterrows(), total=total_rows):
        question = row['question ']
        questions.append(question)
        gt = str(row['answer']).strip()
        if not gt=='':
            references.append(gt)
            predictions.append(query_model(question))

# Evaluate Results
import csv

# Save predictions to file
with open("maroon_chat_predictions.csv", "w") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(["question", "reference", "prediction"])
    for question, reference, prediction in zip(questions, references, predictions):
        csv_writer.writerow([question, reference, prediction])

import numpy as np

bleurt = evaluate.load("bleurt", module_type="metric")
results = bleurt.compute(
    references=references,
    predictions=predictions
)
print(f"BLEU SCORE {np.mean(results['scores'])}")

meteor = evaluate.load("meteor", module_type="metric")
results = meteor.compute(
    references=references,
    predictions=predictions
)
print(results)

BERTSCORE = evaluate.load("bertscore", module_type="metric")
ROUGE = evaluate.load("rouge", module_type="metric")
GLEU = evaluate.load("google_bleu", module_type="metric")
bertscores_scores = BERTSCORE.compute(references=references, predictions=predictions, lang="en", model_type="distilbert-base-uncased")
rouge_scores = ROUGE.compute(references=references, predictions=predictions)
gleu_scores = GLEU.compute(references=references, predictions=predictions)

print("-" * 50)
print("Evaluation metrics summary:")
print("-" * 50)
print("BERTSCORE-precision:", np.mean(bertscores_scores["precision"]))
print("BERTSCORE-recall:", np.mean(bertscores_scores["recall"]))
print("BERTSCORE-f1:", np.mean(bertscores_scores["f1"]))
print("ROUGE-rouge1:", np.mean(rouge_scores["rouge1"]))
print("ROUGE-rouge2:", np.mean(rouge_scores["rouge2"]))
print("ROUGE-rougeL:", np.mean(rouge_scores["rougeL"]))
print("GLEU:", gleu_scores["google_bleu"])
print("-" * 50)