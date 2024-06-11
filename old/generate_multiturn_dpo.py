# Import necessary libraries
import os  # Import the os module for file operations
import json  # Import the json module for JSON operations
import pandas as pd  # Import the pandas library for data manipulation
from vllm import LLM, SamplingParams  # Import the LLM and SamplingParams classes from the vllm module
from transformers import AutoTokenizer, AutoModelForCausalLM  # Import the AutoTokenizer and AutoModelForCausalLM classes from the transformers module
import yaml  # Import the yaml module for YAML operations
import re  # Import the re module for regular expressions
import torch  # Import the torch module for PyTorch operations


# Define the path to the datasets folder
DATASETS_FOLDER = '/raid/ovod/playground/data/jessan/PhytoChat/Webcrawled-info'
# Define the path to the output file
qa_list_file = "/raid/ovod/playground/data/jessan/PhytoChat/data/multiturn_dpo_qa_list.txt"
# Define the model ID
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# Define the system prompt
SYSTEM_PROMPT = "You are a helpful and harmless AI assistant."

# Define a function to generate prompts and extract answers
def generate_prompt(user_query, tokenizer):
    user_instruction = f"""An ideal response to a user query is a response that clarifies if the query does not provide enough information. A rejected response enumerates the causes and the possible solutions to the problem. Generate an ideal and a rejected response to the following user query in YAML format as shown in the example below:
    user_query: "{user_query}"
    ideal_response: "[response]"
    rejected_response: "[response]"
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_instruction},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, truncation=True, max_length=4096)
    return text

# Define a function to generate question-answer pairs
def generate_qa_pairs(prompts, model, sampling_params):
    responses = model.generate(prompts, sampling_params)
    return [response.outputs[0].text for response in responses]


# Create an instance of the LLM model
model = LLM(model_id)

# Create an instance of the AutoTokenizer class
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Create an instance of the SamplingParams class
sampling_params = SamplingParams(temperature=0.75, top_p=0.95, max_tokens=1024)

# Initialize an empty list to store prompts
prompts = []

# Iterate over the files in the datasets folder
with open('data/user_queries.txt', 'r') as f:
    for user_query in f:
        prompts.append(generate_prompt(user_query.strip(), tokenizer))  # Generate prompts and add them to the list

# Generate question-answer pairs using the prompts, model, and sampling parameters
raw_qas = generate_qa_pairs(prompts, model, sampling_params)


# Write the question-answer pairs to the output file
with open(qa_list_file, "w") as f:
    for qa in raw_qas:
        f.write(qa + "\n\n\n")
