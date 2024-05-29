import os
import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

DATASETS_FOLDER = '/home/madee/PhytoChat/webcrawled-info'
SYSTEM_PROMPT = """Generate 10 unique question-answer pairs from the following text, along with a positive response and a negative response for each question. Format: Q: <question> Pos: <positive answer> Neg: <negative answer>"""

def generate_prompt(context, tokenizer):
    """
    Generate a prompt for the LLM model based on the given context.
    """
    messages = [
        {"role": "user", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": "Sure! What is the text that you want to generate questions and answers from?"},
        {"role": "user", "content": context},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, truncation=True, max_length=4096)
    return text

def generate_qa_pairs(prompts, model, sampling_params):
    """
    Generate question-answer pairs with positive and negative responses using the LLM model.
    """
    dataset = []
    try:
        responses = model.generate(prompts, sampling_params)
        for response in responses:
            response = response.outputs[0].text
            response = response.replace('(Positive)', '').replace('(Negative)', '')
            qa_pairs = response.split('\n\n') # Split by double newlines to separate each Q-A pair
            for pair in qa_pairs:
                lines = pair.split('\n')
                prompt, chosen, rejected = None, None, None
                for line in lines:
                    if line.startswith('Q:'):
                        prompt = line[2:].strip()  # Remove 'Q: ' prefix
                    elif line.startswith('Pos:'):
                        chosen = line[5:].strip()  # Remove 'Pos: ' prefix
                    elif line.startswith('Neg:'):
                        rejected = line[5:].strip()  # Remove 'Neg: ' prefix
                if prompt and chosen and rejected:
                    dataset.append({'prompt': prompt, 'chosen': chosen, 'rejected': rejected})
    except Exception as e:
        print(f"Error during model generation: {e}")
    return dataset

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
model = LLM(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
sampling_params = SamplingParams(temperature=0.75, top_p=0.95, max_tokens=1024)

prompts = []
for path, directories, files in os.walk(DATASETS_FOLDER):
    for file in files:
        if file.endswith('.json'):
            with open(os.path.join(path, file), 'r') as f:
                webpages = json.load(f)
                for page in webpages:
                    html_content = page['html']
                    context = html_content.replace('’', "'").replace(' ', ' ')
                    prompts.append(generate_prompt(context, tokenizer))

qa_dataset = generate_qa_pairs(prompts, model, sampling_params)

qa_list_file = "/home/madee/PhytoChat/qa_TRL_dataset.json"
with open(qa_list_file, "w") as f:
    json.dump(qa_dataset, f, indent=4)