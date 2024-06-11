import evaluate
import pandas as pd
import torch
import csv
import json

bleu = evaluate.load("bleu")
meteor = evaluate.load('meteor')

predictions_save_paths = [
    "data/predictions/vanilla_mistral_predictions_sft.json",
    "data/predictions/sft_mistral_predictions_sft.json",
    "data/predictions/dpo_mistral_predictions_sft.json",
    "data/predictions/archer_mistral_predictions_sft.json",
    "data/predictions/vanilla_llama_predictions_sft.json",
    "data/predictions/sft_llama_predictions_sft.json",
    "data/predictions/dpo_llama_predictions_sft.json",
    "data/predictions/archer_llama_predictions_sft.json",
]

print('---')
print('BLEU and METEOR Scores on SFT test data:')
for path in predictions_save_paths:
    print(path)

    with open(path, 'r') as f:
        predictions = json.load(f)

    questions = [sample['question'] for sample in predictions]
    references = [sample['reference'] for sample in predictions]
    predictions = [sample['prediction'] for sample in predictions]

    results = bleu.compute(predictions=predictions, references=references)
    print(f"BLEU: {results['bleu']}")
    results = meteor.compute(predictions=predictions, references=references)
    print(f"METEOR: {results['meteor']}")
    print()


predictions_save_paths = [
    "data/predictions/vanilla_mistral_predictions_dpo.json",
    "data/predictions/sft_mistral_predictions_dpo.json",
    "data/predictions/dpo_mistral_predictions_dpo.json",
    "data/predictions/archer_mistral_predictions_dpo.json",
    "data/predictions/vanilla_llama_predictions_dpo.json",
    "data/predictions/sft_llama_predictions_dpo.json",
    "data/predictions/dpo_llama_predictions_dpo.json",
    "data/predictions/archer_llama_predictions_dpo.json",
]

print('---')
print('BLEU and METEOR Scores on DPO test data:')
for path in predictions_save_paths:
    print(path)

    with open(path, 'r') as f:
        predictions = json.load(f)

    questions = [sample['question'] for sample in predictions]
    references = [sample['reference'] for sample in predictions]
    predictions = [sample['prediction'] for sample in predictions]

    results = bleu.compute(predictions=predictions, references=references)
    print(f"BLEU: {results['bleu']}")
    results = meteor.compute(predictions=predictions, references=references)
    print(f"METEOR: {results['meteor']}")
    print()
