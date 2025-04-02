import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def clean_tweet(tweet):
    """Clean individual tweet text."""
    # Remove URLs
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    # Remove emojis and special characters
    tweet = re.sub(r'[^\w\s,]', '', tweet)  # This removes emojis and special characters
    # Convert to lowercase
    tweet = tweet.lower()
    # Remove extra whitespace
    tweet = tweet.strip()
    return tweet


def process_text(data):
    """Process the tweet text in the DataFrame."""
    # Clean the tweet text
    data['Tweet Text'] = data['Tweet Text'].apply(clean_tweet)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    data['Tweet Text'] = data['Tweet Text'].apply(
        lambda x: ' '.join(word for word in x.split() if word not in stop_words))

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    data['Tweet Text'] = data['Tweet Text'].apply(
        lambda x: ' '.join(lemmatizer.lemmatize(word) for word in x.split()))

    return data
