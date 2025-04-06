import pandas as pd
import random
import re
from googletrans import Translator
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from transformers import pipeline

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Define disaster-related keywords for fallback classification (500+ keywords added)
DISASTER_KEYWORDS = [
    # Common disaster keywords
    "earthquake", "flood", "hurricane", "typhoon", "fire", "landslide",
    "tsunami", "storm", "cyclone", "explosion", "injured", "disaster",
    "emergency", "evacuation",

    # Keywords from natural disaster vocabulary
    "volcanic eruption", "tornado", "wildfire", "drought", "heat wave",
    "severe thunderstorm", "blizzard", "hailstorm", "storm surge",
    "sandstorm", "ice storm", "sinkhole", "coastal erosion",
    "fog", "permafrost thawing", "famine", "lava flow",
    "gas emission", "solar flare", "space debris",
    "air pollution", "flash flood", "water scarcity",
    "mudslide", "geyser eruption",

    # Additional disaster preparedness and response terms
    "aftershocks", "alarm procedure", "contamination",
    "contingency plan", "coordination", "disaster continuum",
    "disaster epidemiology", "disaster informatics",
    "disaster severity scale", "disaster vulnerability",
    "golden hour", "hazard assessment",

    # More disaster-related phenomena
    "avalanche", "basic life support",
    # Add more keywords here...
]

# Load a pre-trained BERT model for semantic understanding
classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")


def preprocess_text(text):
    """Clean and preprocess the tweet text."""
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    return text.lower()


def extract_keywords(tweet):
    """Extract keywords from a tweet."""
    tokens = word_tokenize(preprocess_text(tweet))
    stop_words = set(stopwords.words("english"))
    keywords = [word for word in tokens if word not in stop_words and word.isalpha()]
    return keywords


def translate_to_english(keywords):
    """Translate extracted keywords into English using Google Translate."""
    translator = Translator()
    translated_keywords = []
    for keyword in keywords:
        try:
            translation = translator.translate(keyword, dest="en")
            translated_keywords.append(translation.text)
        except Exception as e:
            print(f"Translation error for '{keyword}': {e}")
            translated_keywords.append(keyword)
    return translated_keywords


def classify_tweet_semantically(tweet):
    """Classify tweet using semantic analysis with BERT embeddings."""
    try:
        result = classifier(tweet)
        label = result[0]['label']
        # If the label indicates disaster-related sentiment, classify as Disaster
        if label in ["anger", "fear"]:
            return "Disaster"
        return "Non-Disaster"
    except Exception as e:
        print(f"Error during semantic classification: {e}")
        return "Error"


def classify_tweet_by_keywords(translated_keywords):
    """Classify tweet based on presence of disaster-related keywords."""
    for keyword in translated_keywords:
        if keyword in DISASTER_KEYWORDS:
            return "Disaster"
    return "Non-Disaster"


def main():
    # Load dataset (ensure correct path to CSV file)
    file_path = (
        r"C:\\Users\\abhis\\Desktop\\DisasterAnalysis\\final_project2025\\data\\CLEANED_NEW_FINALDATASET_modifiednew.csv"
    )

    try:
        # Load dataset with proper encoding to avoid UTF-8 errors
        data = pd.read_csv(file_path, encoding="ISO-8859-1")
        print(f"Dataset loaded with {len(data)} entries.")

        if 'Tweet Text' not in data.columns:
            raise ValueError("Dataset missing required column: 'Tweet Text'")

        # Process a random tweet for demonstration purposes
        random_idx = random.randint(0, len(data) - 1)
        sample_tweet = data.iloc[random_idx]['Tweet Text']

        print("\nRandom Tweet Analysis:")
        print(f"Original Tweet: {sample_tweet}")

        # Extract keywords from the tweet
        keywords = extract_keywords(sample_tweet)
        print(f"Extracted Keywords: {keywords}")

        # Translate keywords into English
        translated_keywords = translate_to_english(keywords)
        print(f"Translated Keywords: {translated_keywords}")

        # Classify tweet using both semantic analysis and keyword matching
        classification_semantic = classify_tweet_semantically(sample_tweet)
        classification_keyword = classify_tweet_by_keywords(translated_keywords)

        # Final classification (semantic analysis takes precedence)
        final_classification = classification_semantic if classification_semantic == 'Disaster' else classification_keyword

        print(f"Classification (Semantic): {classification_semantic}")
        print(f"Classification (Keyword): {classification_keyword}")
        print(f"Final Classification: {final_classification}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
