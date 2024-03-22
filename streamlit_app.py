import streamlit as st
import joblib
import pandas as pd
import numpy as np
from transformers import DistilBertTokenizer, TFDistilBertModel
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('WordNet')
from nltk.stem import WordNetLemmatizer
import tensorflow as tf

# Load sentiment classification model
with open('model/log_classifier_bert.pkl', 'rb') as f:
    log_classifier_bert = joblib.load(f)

# Load DistilBERT model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
distilbert_model = TFDistilBertModel.from_pretrained(model_name)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Define preprocessing function
def preprocess(text):
    sentence = re.sub(r'[^a-zA-Z]', ' ', text)
    sentence = sentence.lower()
    tokens = sentence.split()
    stop_words_excluded = set(stopwords.words("english")) - {'not', 'bad'}
    clean_tokens = [token for token in tokens if token not in stop_words_excluded]
    clean_tokens = [lemmatizer.lemmatize(token) for token in clean_tokens]
    return ' '.join(clean_tokens)

# Function to get DistilBERT embeddings for a given text
def get_distilbert_embeddings(review):
    processed_review = preprocess(review)
    inputs = tokenizer(processed_review, return_tensors="tf", padding=True, truncation=True)
    outputs = distilbert_model(inputs)
    embeddings = outputs.last_hidden_state
    sentence_embedding = tf.reduce_mean(embeddings, axis=1)
    return sentence_embedding.numpy()

# Define the Streamlit app
def main():
    st.title("Sentiment Analysis with DistilBERT")
    review = st.text_input("Enter your review:")
    
    if st.button("Predict"):
        if review:
            processed_review = get_distilbert_embeddings(review)
            sentiment = log_classifier_bert.predict(processed_review)[0]
            if sentiment == "Positive":
                st.success("Positive sentiment")
            else:
                st.error("Negative sentiment")
        else:
            st.warning("Please enter a review")

if __name__ == '__main__':
    main()
