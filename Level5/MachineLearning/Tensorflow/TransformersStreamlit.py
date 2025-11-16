import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch

tokenizergpt = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
gpt = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
tokenizerflan = AutoTokenizer.from_pretrained("google/flan-t5-base")
flan = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
tokenizerblender = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
blenderbot = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")

def generate_text(text,model,tokenizer):
    input = tokenizer.encode(text, return_tensors="pt")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    output = model.generate(input, do_sample=True, pad_token_id=tokenizer.pad_token_id, max_new_tokens=200)
    return tokenizer.decode(output[0], skip_special_tokens=True)

option = st.selectbox(
    "Select Model",
    ("Distilgpt2", "Flan", "Blenderbot"),
)
text_prompt=st.text_input("Type Prompt Here")

if st.button("Generate"):
    if option=="Distilgpt2":
        model=gpt
        st.write(generate_text(text_prompt,model,tokenizergpt))
    if option=="Flan":
        model=flan
        st.write(generate_text(text_prompt,model,tokenizerflan))
    if option=="Blenderbot":
        model=blenderbot
        st.write(generate_text(text_prompt,model,tokenizerblender))