import pandas as pd
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import pyarrow as pa
import time

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# Define a chat template if necessary
# tokenizer.chat_template = ...

# Load Parquet file
parquet_file = '/home/madee/PhytoChat/qna_pairs.parquet'
df = pd.read_parquet(parquet_file)

def process_row(row):
    try:
        question = row['question']
        answer = row['answer']

        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]

        tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        json_object = {
            "text": tokenized_chat,
            "instruction": "You are a friendly chatbot specializing in providing information about the University of the Philippines (UP).",
            "input": question,
            "output": answer
        }

        return json_object
    except Exception as e:
        print(f"Error processing row: {e}")
        return None

# Process DataFrame
start_time = time.time()
json_list = [process_row(row) for index, row in df.iterrows() if process_row(row) is not None]
end_time = time.time()

new_df = pd.DataFrame(json_list)

# Split the dataset
train_df, test_df = train_test_split(new_df, test_size=0.3, random_state=42)

# Save to Parquet
train_df.to_parquet('/home/madee/PhytoChat/train_df.parquet', index=False)
test_df.to_parquet('/home/madee/PhytoChat/test_df.parquet', index=False)

# Compute duration
duration = end_time - start_time
print("Duration:", duration, "seconds")