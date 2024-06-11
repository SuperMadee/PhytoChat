# Import necessary libraries
import os  # Import the os module for file operations
import json  # Import the json module for JSON operations
import pandas as pd  # Import the pandas library for data manipulation
from vllm import LLM, SamplingParams  # Import the LLM and SamplingParams classes from the vllm module
from transformers import AutoTokenizer, AutoModelForCausalLM  # Import the AutoTokenizer and AutoModelForCausalLM classes from the transformers module
import yaml  # Import the yaml module for YAML operations
import re  # Import the re module for regular expressions
import torch  # Import the torch module for PyTorch operations
import pandas as pd
import re
import time
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import pyarrow as pa
import time

os.environ['HF_TOKEN'] = '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'
# Define the model ID
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# Define the path to the output file
qa_list_file = "data/sft/qa_list.txt"
os.makedirs(os.path.dirname(qa_list_file), exist_ok=True)
# Define the path to the datasets folder
DATASETS_FOLDER = 'data/crawled'

# Define the system prompt
SYSTEM_PROMPT = """Generate 20 unique question-answer pairs that can easily be understood by farmers from the following text. Format: Q: <question> A: <answer>"""

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

def main():
    # Create an instance of the LLM model
    model = LLM(model_id)

    # Create an instance of the AutoTokenizer class
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

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

    # Write the question-answer pairs to the output file
    with open(qa_list_file, "w") as f:
        for qa in raw_qas:
            f.write(qa + "\n\n\n")

    with open(qa_list_file, 'r') as f:
        text = f.read()

    # Split the text into sections using regular expressions
    sections = re.split(r'\n\s*\n', text.strip())

    # Measure the time taken for the whole process
    start_time = time.time()

    # Initialize an empty list to store Q&A pairs
    qna_pairs = []
    for section in sections:
        qna_pairs.extend(extract_qa_pairs(section))

    # Filter out incomplete Q&A pairs
    qna_pairs = [pair for pair in qna_pairs if pair["question"] and pair["answer"]]

    # Convert the Q&A pairs to a DataFrame
    df = pd.DataFrame(qna_pairs)

    # Save the DataFrame as a CSV file
    # df.to_csv('/home/madee/PhytoChat/qna_pairs.csv', index=False)

    # Convert the DataFrame to an Arrow Table
    table = pa.Table.from_pandas(df)

    # Process DataFrame
    json_list = [process_row(row) for index, row in df.iterrows() if process_row(row) is not None]

    new_df = pd.DataFrame(json_list)

    # Split the dataset
    train_df, test_df = train_test_split(new_df, test_size=0.1)
    validation_df, test_df = train_test_split(test_df, test_size=0.5)

    # Save to Parquet
    train_df.to_parquet('data/sft/train.parquet', index=False)
    validation_df.to_parquet('data/sft/validation.parquet', index=False)
    test_df.to_parquet('data/sft/test.parquet', index=False)

    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(validation_df)}")
    print(f"Test set size: {len(test_df)}")


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

def extract_qa_pairs(text):
    qa_pairs = []
    # Update the regular expression to remove additional words
    text = re.sub(r'^[0-9]+[.]|q+[:]|a+[:]|Q+[0-9]*[:]|A+[0-9]*[:]|Question+[:]|Answer+[:]|"|', '', text, flags=re.MULTILINE).strip()
    # Split the text into paragraphs
    paragraphs = text.strip().split('\n')
    # Filter out empty lines
    paragraphs = [p for p in paragraphs if p.strip()]
    
    # Temporary storage for the question
    question_temp = None
    # Iterate through paragraphs to assign questions and answers
    for paragraph in paragraphs:
        # Check if the paragraph ends with a question mark
        if paragraph.strip().endswith('?'):
            question_temp = paragraph.strip()
        else:
            # Check if there's a question waiting for an answer
            if question_temp:
                # Append the Q&A pair
                qa_pairs.append({"question": question_temp, "answer": paragraph.strip()})
                # Reset the temp question
                question_temp = None
    return qa_pairs

def process_row(row):
    try:
        question = row['question']
        answer = row['answer']

        messages = [
            {"role": "system", "content": "You are a helpful and harmless AI plant care assistant."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]

        tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=False)
        
        json_object = {
            "text": tokenized_chat,
            "question": question,
            "answer": answer
        }

        return json_object
    except Exception as e:
        print(f"Error processing row: {e}")
        return None
    

if __name__ == "__main__":
    main()