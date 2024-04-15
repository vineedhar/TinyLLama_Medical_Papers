import streamlit as st
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# Load your model and tokenizer from Hugging Face
model_name = "Vineedhar/Medical_papers_inference_TinyLLama"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the pipeline with your model
pipe = pipeline("Chatbot", model=model, tokenizer=tokenizer)

text = st.text_area("Enter some text:")

if text:
    out = pipe(text)
    st.json(out)