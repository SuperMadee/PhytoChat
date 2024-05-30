import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

from tqdm import tqdm
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer, pipeline, BitsAndBytesConfig  # Importing various classes from the transformers module
import torch
from datasets import load_dataset


JSON_DATA_PATH = '/raid/ovod/playground/data/jessan/PhytoChat/data/qa_TRL_dataset.json'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


config = PPOConfig(
    learning_rate=1.41e-5,
)

model_id = 'meta-llama/Meta-Llama-3-8B-Instruct'


dataset = load_dataset('json', data_files=JSON_DATA_PATH, split='train')
dataset = dataset.rename_column("prompt", "query")


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True
)
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map='auto',
    torch_dtype=torch.bfloat16,
    attn_implementation='flash_attention_2',
    cache_dir='/raid/ovod/playground/data/.cache/huggingface/hub'
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

reward_model = pipeline("text-classification", model="distilbert/distilbert-base-uncased", device_map={'':0})

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}

def tokenize(sample):
    sample["input_ids"] = tokenizer(sample['query'], padding='max_length', truncation=True, return_tensors='pt', max_length=64)['input_ids']
    return sample

dataset = dataset.map(tokenize, batched=False)
ppo_trainer = PPOTrainer(
    model=model,
    config=config,
    dataset=dataset,
    tokenizer=tokenizer,
)

epochs = 3
for epoch in tqdm(range(epochs), "epoch: "):
    for batch in tqdm(ppo_trainer.dataloader): 
        query_tensors = batch['input_ids'][0]
    
        #### Get response from SFTModel
        response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
    
        #### Compute reward score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = reward_model(texts)
        rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
    
        #### Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

#### Save model
ppo_trainer.save_pretrained("merged-model-8b")
