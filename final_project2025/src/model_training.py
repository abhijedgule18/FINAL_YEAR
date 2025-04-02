import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


def train_model(data):
    """Train machine learning models on the tweet data."""

    # Vectorize the tweet text using TF-IDF vectorizer.
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['Tweet Text'])

    # Encode the target variable (Information Type).
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data['Information Type'])

    # Split into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost Classifier.
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    xgb_model.fit(X_train, y_train)

    # Train LightGBM Classifier.
    lgb_model = LGBMClassifier()
    lgb_model.fit(X_train, y_train)

    # Evaluate models.
    for model, name in zip([xgb_model, lgb_model], ['XGBoost', 'LightGBM']):
        y_pred = model.predict(X_test)
        print(f"Classification Report for {name}:")
        print(classification_report(y_test, y_pred,
                                    target_names=label_encoder.classes_))

    return xgb_model, lgb_model, vectorizer, label_encoder  # Return models and encoders.
