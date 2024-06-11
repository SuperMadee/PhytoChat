import os
import glob
import random
import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
import pyarrow as pa

crawled_folder = 'data/crawled'
conversations_path = "data/dpo/conversations.json"

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

system_prompt = "You are a helpful and harmless AI assistant."
user_instruction = """Simulate a multi-turn conversation from the following text between PhytoChat, a plant care assistant, and a user who is concerned about the health of his plants. Follow the following format:
User:
PhytoChat:
User:
PhytoChat:
"""

def main():
    model = LLM(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    sampling_params = SamplingParams(
        temperature=0.75, 
        top_p=0.95, 
        max_tokens=2048
    )

    prompts = []
    paths = glob.glob(crawled_folder + '/*.json')
    print(f'Found {len(paths)} .json files in ./data/crawled')
    for path in paths:
            with open(path, 'r') as f:
                pages = json.load(f)
                for page in pages:
                    content = page['html']
                    context = content.replace('’', "'").replace(' ', '')
                    prompts.append(generate_prompt(context, tokenizer))
    
    conversations = generate_conversations(prompts, model, sampling_params)
    
    os.makedirs(os.path.dirname(conversations_path), exist_ok=True)
    with open(conversations_path, 'w') as f:
        json.dump(conversations, f, indent=4)
    


# Define a function to generate prompts and extract answers
def generate_prompt(context, tokenizer):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_instruction},
        {"role": "assistant", "content": "Sure! What is the text that you want to generate questions and answers from?"},
        {"role": "user", "content": "Text:\n" + context}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, truncation=True, max_length=4096)
    return text

def generate_conversations(prompts, model, sampling_params):
    responses = model.generate(prompts, sampling_params)
    responses = [response.outputs[0].text for response in responses]

    conversations = []
    for response in responses:
        conversation = [{'role': 'system', 'content': 'You are PhytoChat, a helpful and harmless AI plant care assistant.'}]

        for line in response.split('\n'):
            if line.lower().startswith('user:'):
                line = line[5:].strip() 
                if line != '':
                    conversation.append({'role': 'user', 'content': line})
            elif line.lower().startswith('phytochat:'):
                line = line[10:].strip() 
                if line != '':
                    conversation.append({'role': 'assistant', 'content': line})

        if len(conversation) >= 3:
            conversations.append({'conversation': conversation})

    return conversations


if __name__ == '__main__':
    main()