import pandas as pd
import re
import time
import pyarrow as pa
import pyarrow.parquet as pq

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

# Read the text file
with open('/home/madee/PhytoChat/qa_list.txt', 'r') as f:
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
df.to_csv('/home/madee/PhytoChat/qna_pairs.csv', index=False)

# Convert the DataFrame to an Arrow Table
table = pa.Table.from_pandas(df)

# Write the Arrow Table to a Parquet file
pq.write_table(table, '/home/madee/PhytoChat/qna_pairs.parquet')

# Calculate the elapsed time
elapsed_time = time.time() - start_time
print("Elapsed time:", elapsed_time, "seconds")