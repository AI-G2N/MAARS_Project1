import os
import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
from googletrans import Translator
from dotenv import load_dotenv
from huggingface_hub import login

# Load environment variables from .env file
load_dotenv()

# Get Hugging Face API key from environment variable
huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY')

# Authenticate with Hugging Face Hub
if huggingface_api_key:
    login(token=huggingface_api_key)
else:
    st.error("Hugging Face API key not found. Please add it to the .env file.")

# Initialize the Google Translator client
def initialize_translation_client():
    return Translator()

# Translate text using Googletrans
def translate_text(client, text, target_language='en'):
    try:
        translation = client.translate(text, dest=target_language)
        return translation.text
    except Exception as e:
        st.error(f"Translation failed: {e}")
        return text  # Return the original text if translation fails

# Function to check if a sentence is in English
def is_english(client, sentence):
    try:
        detection = client.detect(sentence)
        return detection.lang == 'en'
    except Exception as e:
        st.error(f"Language detection failed: {e}")
        return True  # Assume the text is in English if detection fails

# Function to check toxicity
def check_toxicity(sentences):
    results = []
    for sentence in sentences:
        analysis = toxic_pipeline(sentence)
        is_toxic = analysis[0]['label'] == 'toxic'
        toxic_score = int(analysis[0]['score'] * 100) if is_toxic else 0
        results.append((sentence, is_toxic, toxic_score))
    return results

# Function to perform sentiment analysis
def analyze_sentiment(sentences):
    results = []
    analysis_results = sentiment_pipeline(sentences)
    for sentence, analysis in zip(sentences, analysis_results):
        sentiment = analysis['label']
        results.append((sentence, sentiment))
    return results

# Streamlit UI
st.title("Feedback Analysis")

# Input for sentences
sentences_input = st.text_area("Enter sentences (separate by new lines):")
sentences = sentences_input.split('\n')

# Initialize translation client
translation_client = initialize_translation_client()

# Load the toxic comment model and tokenizer
toxic_model_path = "martin-ha/toxic-comment-model"
toxic_tokenizer = AutoTokenizer.from_pretrained(toxic_model_path, revision="main")
toxic_model = AutoModelForSequenceClassification.from_pretrained(toxic_model_path, revision="main")
toxic_pipeline = TextClassificationPipeline(model=toxic_model, tokenizer=toxic_tokenizer)

# Load the sentiment analysis model and tokenizer
sentiment_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name, revision="main")
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name, revision="main")
sentiment_pipeline = TextClassificationPipeline(model=sentiment_model, tokenizer=sentiment_tokenizer)

# Analyze button
if st.button("Analyze"):
    if sentences:
        # Check if sentences are in English, translate if necessary
        translated_sentences = []
        for sentence in sentences:
            if sentence.strip():  # Skip empty lines
                if not is_english(translation_client, sentence):
                    translated_sentence = translate_text(translation_client, sentence, 'en')
                    translated_sentences.append(translated_sentence)
                else:
                    translated_sentences.append(sentence)

        # Perform toxicity check
        toxicity_results = check_toxicity(translated_sentences)

        toxic_comments = []
        non_toxic_sentences = []

        for sentence, is_toxic, toxic_score in toxicity_results:
            if is_toxic:
                toxic_comments.append((sentence, toxic_score))
            else:
                non_toxic_sentences.append(sentence)

        # Perform sentiment analysis on non-toxic sentences
        sentiment_results = analyze_sentiment(non_toxic_sentences)

        # Display results
        if toxic_comments:
            st.write("Toxic Comments:")
            for sentence, toxic_score in toxic_comments:
                st.write(f"Sentence: {sentence}")
                st.write(f"Toxic Score: {toxic_score}%")
                st.write("Comment has toxic content. Please rewrite the comment.")
                st.write("---")

        if sentiment_results:
            st.write("Sentiment Analysis Results:")
            for sentence, sentiment in sentiment_results:
                st.write(f"Sentence: {sentence}")
                st.write(f"Sentiment: {sentiment}")
                st.write("---")
    else:
        st.write("Please enter at least one sentence.")
