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
DATASETS_FOLDER = 'data/crawled'
# Define the path to the output file
qa_list_file = "data/dpo/multi_turn_data.json"
# Define the model ID
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# Define the system prompt
SYSTEM_PROMPT = "You are a helpful and harmless AI assistant."

# Define a function to generate prompts and extract answers
def generate_prompt(user_query, tokenizer):
    user_instruction = f"""An ideal response to a user query is a response that asks the user to clarify if the query does not provide enough information. A rejected response enumerates the causes and the possible solutions to the problem. Generate an ideal and a rejected response to the following user query in YAML format as shown in the example below:
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
def generate_chosen_rejected_responses(prompts, model, sampling_params):
    responses = model.generate(prompts, sampling_params)
    return [response.outputs[0].text for response in responses]

# Create an instance of the LLM model
model = LLM(model_id, dtype=torch.bfloat16)

# Create an instance of the AutoTokenizer class
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Create an instance of the SamplingParams class
sampling_params = SamplingParams(temperature=0.75, top_p=0.95, max_tokens=1024)

# Initialize an empty list to store prompts
prompts = []

with open('data/dpo/conversations.json', 'r') as f:
    conversations = json.load(f)
    queries = list(set([conversation['conversation'][1]['content'] for conversation in conversations]))

for query in queries:
    prompts.append(generate_prompt(query.strip(), tokenizer))  # Generate prompts and add them to the list

# Generate question-answer pairs using the prompts, model, and sampling parameters
responses = generate_chosen_rejected_responses(prompts, model, sampling_params)

data = []
for query, response in zip(queries, responses):
    # Make indentation consistent
    response = 'ideal_response' + response.split('ideal_response')[1]
    response = response.replace('\n    ', '\n')
    response = response.replace('\n  ', '\n')
    response = response.replace('\n ', '\n')
    response = response.replace('\n', '\n    ')

    # response = response.replace(':', '(COLON)')
    response = response.replace('- ', '* ')
    # Unindent start of responses
    response = response.replace('\n    rejected_response:', '\nrejected_response:')
    response = response.replace('\n    ideal_response:', '\nideal_response:')

    try:
        row = yaml.safe_load(response)
    except Exception as e:
        print(f"Exception: {e}")
        print(f"Response: {response}")
        continue

    row_copy = row.copy()
    for key in row_copy.keys():
        if key != 'ideal_response' and key != 'rejected_response':
            row.pop(key)
    
    row['user_query'] = query

    if len(row) == 3:
        data.append(row)

data2 = []
for d in data:
    instance = {}
    instance['prompt'] = d['user_query']
    instance['chosen'] = d['ideal_response']
    instance['rejected'] = d['rejected_response']
    data2.append(instance)

# Write the question-answer pairs to the output file
with open(qa_list_file, "w") as f:
    json.dump(data2, f, indent=4)
