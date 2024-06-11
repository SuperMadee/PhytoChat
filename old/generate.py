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
DATASETS_FOLDER = '/home/madee/PhytoChat/webcrawled-info'

# Define the system prompt
SYSTEM_PROMPT = """Generate 20 unique question-answer pairs that can easily be understood by farmers from the following text. Format: Q: <question> A: <answer>"""

# Define a function to generate prompts and extract answers
def generate_prompt(context, tokenizer):
    messages = [
        {"role": "user", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": "Sure! What is the text that you want to generate questions and answers from?"},
        {"role": "user", "content": context},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, truncation=True, max_length=4096)
    return text

# Define a function to generate question-answer pairs
def generate_qa_pairs(prompts, model, sampling_params):
    responses = model.generate(prompts, sampling_params)
    return [response.outputs[0].text for response in responses]

# Define the model ID
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# Create an instance of the LLM model
model = LLM(model_id)

# Create an instance of the AutoTokenizer class
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Create an instance of the SamplingParams class
sampling_params = SamplingParams(temperature=0.75, top_p=0.95, max_tokens=1024)

# Initialize an empty list to store prompts
prompts = []

# Iterate over the files in the datasets folder
for path, directories, files in os.walk(DATASETS_FOLDER):
    for file in files:
        if file.endswith('.json'):
            with open(os.path.join(path, file), 'r') as f:
                webpages = json.load(f)
                for page in webpages:
                    html_content = page['html']
                    context = html_content.replace('’', "'").replace(' ', '')  # Clean the HTML content
                    prompts.append(generate_prompt(context, tokenizer))  # Generate prompts and add them to the list

# Generate question-answer pairs using the prompts, model, and sampling parameters
raw_qas = generate_qa_pairs(prompts, model, sampling_params)

# Define the path to the output file
qa_list_file = "/home/madee/PhytoChat/qa_list.txt"

# Write the question-answer pairs to the output file
with open(qa_list_file, "w") as f:
    for qa in raw_qas:
        f.write(qa + "\n\n\n")