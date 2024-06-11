# Import the necessary libraries
import gradio as gr
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM
import torch
import accelerate
import time
import random
import emoji

# Set the directory where the model is stored
model_dir = "/home/madee/PhytoChat/merged_model-8B"

# Initialize the tokenizer and model using the directory
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoPeftModelForCausalLM.from_pretrained(model_dir, device_map={"":0}, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, return_dict=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Define a function to generate instructions based on a prompt
def generate_instructions(prompt):

   # Encode the prompt using the tokenizer
    messages = [{"role": "system", "content": "You are a PhytoChat and you are only allowed to answer questions related to plants"},
                {"role": "user", "content": prompt}]
    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(tokenized_chat, return_tensors="pt")

    if torch.cuda.is_available():
        inputs = inputs.to('cuda')

    # Generate results
    outputs = model.generate(
        **inputs,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=200,
        max_length=200,
    )
    decoded_outputs = tokenizer.batch_decode(outputs, clean_up_tokenization=True)
    print(decoded_outputs[0])
    result = decoded_outputs[0].split("assistant<|end_header_id|>")[1].split("<|eot_id|>")[0].strip()
    return result

def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)

# Function to handle the retry action
def retry_action(chat_history):
    if chat_history:
        message = chat_history[-1][0]  # Get the last user message
        return respond(message, chat_history[:-1])  # Resend the last message and remove the last bot response
    return "", chat_history

# Function to handle the undo action
def undo_action(chat_history):
    if chat_history:
        return "", chat_history[:-2]  # Remove the last user-bot message pair
    return "", chat_history

css = """
<style>
button { width: auto; }
</style>
"""

with gr.Blocks() as demo:
    gr.Markdown("<center><span style='font-size: 24px;'>🌱 PhytoChat 🌱</span></center>")
    gr.Markdown("<center>Hi! I am PhytoChat, a chatbot trained on LLama-3 8B parameters. I can answer questions about your plant's health.</center>")

    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    # clear = gr.ClearButton([msg, chatbot])
    # clear.label = emoji.emojize(":wastebasket: Clear")

    def respond(message, chat_history):
        bot_message = generate_instructions(message)
        chat_history.append((message, bot_message))
        time.sleep(2)
        return "", chat_history

    # submit = gr.Button("Submit")
    # submit.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])

    # retry = gr.Button("Retry")
    # retry.click(retry_action, inputs=[chatbot], outputs=[msg, chatbot])

    # undo = gr.Button("Undo")
    # undo.click(undo_action, inputs=[chatbot], outputs=[msg, chatbot])


    # Arrange buttons in a row
    with gr.Row():
        submit = gr.Button("Submit")
        submit.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
        
        retry = gr.Button("Retry")
        retry.click(retry_action, inputs=[chatbot], outputs=[msg, chatbot])
        
        # undo = gr.Button("Undo")
        # undo.click(undo_action, inputs=[chatbot], outputs=[msg, chatbot])
        
        clear = gr.ClearButton([msg, chatbot])
    
    # Update chatbot like functionality
    chatbot.like(print_like_dislike)
    
demo.launch(server_port=8008, server_name='0.0.0.0', share=True)