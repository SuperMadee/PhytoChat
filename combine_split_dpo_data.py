import random
import json

test_len = 2

with open('data/dpo/qa_TRL_dataset.json', 'r') as f:
    data = json.load(f)

with open('data/dpo/multi_turn_data.json', 'r') as f:
    data2 = json.load(f)

data = data + data2
random.shuffle(data)

with open('data/dpo/combined_dpo_dataset.json', 'w') as f:
    json.dump(data, f, indent=4)

with open('data/dpo/combined_dpo_dataset.json', 'r') as f:
    data = json.load(f)

random.shuffle(data)
test = data[:test_len]
validation = data[test_len:test_len*2]
train = data[test_len*2:]

with open('data/dpo/dpo_train.json', 'w') as f:
    json.dump(train, f, indent=4)

with open('data/dpo/dpo_validation.json', 'w') as f:
    json.dump(validation, f, indent=4)

with open('data/dpo/dpo_test.json', 'w') as f:
    json.dump(test, f, indent=4)

import pandas as pd

df = pd.read_json('data/dpo/dpo_test.json')
df.to_parquet('data/dpo/test.parquet')