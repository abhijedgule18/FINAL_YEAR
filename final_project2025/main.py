import pandas as pd
import random
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from googletrans import Translator
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load the pre-trained DistilBERT model and tokenizer for disaster classification
MODEL_NAME = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)


def preprocess_text(text):
    """Preprocess the input text by removing URLs, mentions, and punctuation."""
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.lower()


def classify_tweet(tweet):
    """Classify a tweet as disaster or non-disaster."""
    try:
        # Tokenize the input tweet
        inputs = tokenizer(tweet, return_tensors="pt", truncation=True, padding=True, max_length=512)

        # Get the model's predictions
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Extract confidence scores and predicted class
        confidence, pred_class = torch.max(probs, dim=-1)

        # Map predictions to labels (assuming binary classification here)
        labels = ["Non-Disaster", "Disaster"]
        prediction = labels[pred_class.item()]

        return prediction, confidence.item()
    except Exception as e:
        print(f"Error during classification: {e}")
        return "Error", 0.0


def extract_keywords(tweet):
    """Extract keywords from a tweet."""
    tokens = word_tokenize(preprocess_text(tweet))
    stop_words = set(stopwords.words('english'))
    keywords = [word for word in tokens if word not in stop_words and word.isalpha()]
    return keywords


def translate_to_english(keywords):
    """Translate extracted keywords into English using Google Translate."""
    translator = Translator()
    translated_keywords = []

    for keyword in keywords:
        try:
            translation = translator.translate(keyword, src='auto', dest='en')
            translated_keywords.append(translation.text)
        except Exception as e:
            print(f"Translation error for '{keyword}': {e}")
            translated_keywords.append(keyword)

    return translated_keywords


def main():
    # Load your dataset (make sure to adjust the path)
    file_path = "data/CLEANED_NEW_FINALDATASET.csv"

    try:
        data = pd.read_csv(file_path)  # Load the dataset
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Ensure there are tweets in the dataset before selecting a random one
    if data.empty or 'Tweet Text' not in data.columns:
        print("The dataset is empty or does not contain 'Tweet Text' column.")
        return

    # Select a random tweet from the dataset
    random_index = random.randint(0, len(data) - 1)
    random_tweet = data['Tweet Text'].iloc[random_index]  # Adjust column name if necessary

    print(f"\nRandom Tweet: {random_tweet}")

    # Predict the disaster type for the selected tweet
    prediction, confidence = classify_tweet(random_tweet)

    print(f"Prediction: {prediction} (Confidence: {confidence:.2f})")

    if prediction == "Disaster":
        # Extract keywords and translate them into English
        keywords = extract_keywords(random_tweet)
        translated_keywords = translate_to_english(keywords)

        print(f"Extracted Keywords: {keywords}")
        print(f"Translated Keywords: {translated_keywords}")
    else:
        print("Prediction: Non-Disaster")


if __name__ == "__main__":
    main()
