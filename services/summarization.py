from flask import Flask, request, render_template
from transformers import BartForConditionalGeneration, BartTokenizer
import spacy


# Load BART model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Load spaCy model for keyword extraction
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    # Simplify specific phrases in the text
    return text.replace('which makes it easier', 'which simplifies')

def postprocess_summary(summary):
    # Ensure summary ends with punctuation
    if not summary.endswith(('.', '!', '?')):
        summary = summary.rsplit('.', 1)[0] + '.'
    return summary

def chunk_text(text, max_chunk_size=1024):
    # Split text into chunks that fit within the model's max token limit
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def extract_keywords(text):
    # Extract keywords using spaCy
    doc = nlp(text)
    return " ".join([chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1])

def summarize_text(text):
    preprocessed_text = preprocess_text(text)
    keywords = extract_keywords(preprocessed_text)
    chunks = chunk_text(preprocessed_text)
    
    summaries = []
    for chunk in chunks:
        inputs = tokenizer.encode("summarize: " + keywords + " " + chunk, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(
            inputs,
            max_length=124,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    
    final_summary = " ".join(summaries)
    return postprocess_summary(final_summary)