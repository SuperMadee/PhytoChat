import random
import json
from typing import Optional, Dict
import time
from openai import OpenAI
import logging
logging.getLogger().setLevel(logging.CRITICAL)
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import concurrent.futures
from termcolor import cprint
# openai.util.logger.setLevel(logging.WARNING)

TEMPLATE = """<s>[INST]You are User, a plant grower who is concerned with the health of your plant. In one to two sentences, respond to PhytoChat by providing more information about your plant. Say thank you if the disease is diagnosed and you receive a solution.

An example conversation is as follows:
PhytoChat: Hello! How can I help you today?
User: My plant has yellow spots on its leaves. What should I do?
PhytoChat: May I ask if the spots are on the upper or lower side of the leaves?
User: They are on the upper side.
PhytoChat: The yellow spots on the upper side of the leaves may indicate a fungal infection. You can try removing the affected leaves and applying a fungicide.
User: Thank you, PhytoChat!

Please continue this conversation by responding to PhytoChat as User.
{obs}

Please answer in the following format:
{
"Response": "Your Response",
}[/INST]"""

def mistral_city_decode_actions(output):
    """
    Decode the actions from the output of the model.
    """
    actions = []
    for a in output:
        action = a.split('"Response":')[-1]
        action = action.split("}")[0].strip()
        action = action.strip().replace('"', '')
        actions.append(action)
    return actions

concerns = [
    "My plant has yellow spots on its leaves. Can you help me?",
    "I think my plant has a fungal infection.",
    "My plant has black spots on its leaves. What should I do?",
]

with open("/raid/ovod/playground/data/jessan/phytochat/data/dpo/multi_turn_data.json") as f:
    data = json.load(f)
    concerns = list(set([d['prompt'] for d in data]))
    print(f"Number of concerns: {len(concerns)}")

INITIAL_STR = f"""PhytoChat: Hello, how can I help you today?
User: {random.choice(concerns)}
"""

class PhytoChatEnv():
    def __init__(
        self, 
        # word_list,  
        max_conversation_length: int=20,
    ):
        self.max_conversation_length = max_conversation_length
        self.random = random.Random(None)
        self.count = 0
        self.history = ''
        self.done = True

    # def is_correct(self, question):
    #     #check for the last word
    #     # cut out punctuations at the end
    #     while len(question) > 0 and not question[-1].isalpha():
    #         question = question[:-1]

    #     if len(question) == 0:
    #         return False
    #     # this is the name of the city
    #     word = self.curr_word.lower().split(",")[0]
    #     return word in question.lower()
    #     # guess = question.split(" ")[-1].lower()
    #     # return guess in self.curr_word.lower().split(",")[0] and len(guess) >= 3

    def is_correct(self):
        user_response = self.history.split("User:")[-1].strip().lower()
        return 'thank' in user_response

    def _step(self, question, answer):
        answer = answer.split('?')[-1].strip() # Remove prompt from output
        if self.done:
            return None
        # if self.curr_word.lower().split(",")[0] in answer.lower():
        #     answer = "I can't answer that question."
        self.count+=1
        # self.history += question + ' ' + answer + '\n'
        self.history += f"PhytoChat: {question}\nUser: {answer}\n"
        done = self.is_correct()
        
        if 6 >= self.count > 3:
            reward = 1
        else:
            reward = -1
        
        #if correct reward is -1
        if done:
            reward = 2
        
        self.done = done or self.count == self.max_conversation_length
        return  self.history, reward, self.done
        
    def reset(self, idx : Optional[int]=None):
        self.count = 0 
        self.history = INITIAL_STR 
        self.done = False
        return INITIAL_STR
        # return (Text(INITIAL_STR, is_action=False),)


class BatchedPhytoChatEnv():
    def __init__(
        self, 
        env_load_path: str,
        device,
        cache_dir: str,
        max_conversation_length: int=5,
        bsize: int=4,
    ):
        self.env_list = [PhytoChatEnv(max_conversation_length) for _ in range(bsize)]
        self.bsize = bsize

        # # old
        # self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small", cache_dir=cache_dir)
        # self.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small", cache_dir=cache_dir).to(device)
        # self.model.load_state_dict(torch.load(env_load_path)['model_state_dict'])

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2",
            quantization_config=bnb_config,
            device_map='auto',
            torch_dtype=torch.bfloat16,
            cache_dir="/raid/ovod/playground/data/.cache/huggingface/hub",
            attn_implementation='flash_attention_2'
        )
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    def generate_answers(self, questions):
        histories = [env.history for env in self.env_list]
        inputs = [TEMPLATE.replace('{obs}', history + f'PhytoChat: {question}') for history, question in zip(histories, questions)]
        encoder_ids = self.tokenizer(inputs ,padding=True, return_tensors='pt').to(self.model.device)
        outputs = self.tokenizer.batch_decode(self.model.generate(input_ids=encoder_ids['input_ids'], attention_mask=encoder_ids['attention_mask'],\
                                                                max_new_tokens=128, do_sample = False), skip_special_tokens= True)
        answers = mistral_city_decode_actions(outputs)
        
        # for input, answer in zip(inputs, answers):
        #     print(f"INPUT: {input}")
        #     print(f"ANSWER: {answer}")
        #     print()
        return answers

    def reset(self, idx: Optional[int] = None):
        return [env.reset(idx) for env in self.env_list]
    
    def step(self, questions):
        answers = self.generate_answers(questions)
        # print("Step once!")
        with concurrent.futures.ThreadPoolExecutor() as executor: 
            jobs = [executor.submit(env._step, q, a) for env, q, a in zip(self.env_list, questions, answers)]
            results = [job.result() for job in jobs]
        return results
