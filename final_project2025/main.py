# main.py
import pandas as pd
import random
import re
from googletrans import Translator
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from src.data_cleaning import load_data, clean_data
from src.text_processing import process_text
from src.visualization import visualize_data

import torch
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Define disaster-related keywords
DISASTER_KEYWORDS = [
    "earthquake", "flood", "hurricane", "typhoon", "fire", "landslide",
    "tsunami", "storm", "cyclone", "explosion", "injured", "disaster",
    "emergency", "evacuation", "volcanic eruption", "tornado", "wildfire",
    "drought", "heat wave", "severe thunderstorm", "blizzard", "hailstorm",
    "storm surge", "sandstorm", "ice storm", "sinkhole", "coastal erosion",
    "fog", "permafrost thawing", "famine", "lava flow", "gas emission",
    "solar flare", "space debris", "air pollution", "flash flood",
    "water scarcity", "mudslide", "geyser eruption", "aftershocks",
    "alarm procedure", "contamination", "contingency plan", "coordination",
    "disaster continuum", "disaster epidemiology", "disaster informatics",
    "disaster severity scale", "disaster vulnerability", "golden hour",
    "hazard assessment", "avalanche", "basic life support", "collapse",
    "outbreak", "radiation", "biological hazard", "chemical spill",
    "infrastructure failure", "power outage", "blackout", "water contamination",
    "food shortage", "mass displacement", "refugees", "casualties",
    "fatalities", "missing persons", "search and rescue", "relief efforts",
    "humanitarian aid", "crisis management", "risk assessment", "preparedness",
    "resilience", "impact assessment", "recovery", "reconstruction",
    "climate change", "environmental degradation", "deforestation", "pollution",
    "toxic", "hazardous", "radiation leak", "nuclear accident",
    "industrial accident", "building collapse", "bridge collapse", "dam failure",
    "levee failure", "pipe burst", "gas leak", "oil spill", "train derailment",
    "plane crash", "shipwreck", "terrorism", "civil unrest", "war", "conflict",
    "riot", "looting", "arson", "sabotage", "attack", "bomb", "shooting",
    "hostage", "cyberattack", "data breach", "information warfare",
    "propaganda", "misinformation", "rumors", "panic", "fear", "anxiety",
    "stress", "trauma", "mental health", "psychological support", "counseling",
    "grief", "loss", "bereavement", "community support", "social services",
    "vulnerable populations", "elderly", "children", "disabled", "homeless",
    "low-income", "marginalized", "aid distribution", "shelter", "supplies",
    "medical assistance", "first aid", "triage", "hospital", "clinic",
    "disease control", "vaccination", "public health", "epidemic", "pandemic",
    "quarantine", "lockdown", "social distancing", "mask", "sanitizer",
    "hygiene", "prevention", "preparedness drills", "emergency response plan",
    "mutual aid", "volunteer", "donation", "fundraising", "awareness campaign",
    "public service announcement", "information dissemination", "communication",
    "media coverage", "press conference", "government response", "policy",
    "regulation", "law enforcement", "military", "national guard",
    "international aid", "foreign assistance", "united nations", "red cross",
    "non-governmental organization", "ngo", "community organization",
    "grassroots movement", "citizen action", "social responsibility",
    "ethical considerations", "human rights", "dignity", "compassion",
    "solidarity", "unity", "cooperation", "collaboration", "partnership",
    "sustainable development", "environmental protection", "climate resilience",
    "urban planning", "infrastructure development", "economic recovery",
    "social justice", "equality", "inclusion", "empowerment", "participation",
    "transparency", "accountability", "governance", "leadership", "innovation",
    "technology", "data analysis", "mapping", "gis", "remote sensing",
    "early warning system", "monitoring", "surveillance", "research",
    "scientific assessment", "evidence-based decision making", "best practices",
    "lessons learned", "continuous improvement", "adaptation", "transformation",
    "building back better", "future preparedness", "global cooperation",
    "planetary health", "human security", "peace", "security", "safety",
    "well-being", "flammable", "radioactive", "asphyxiation", "biohazard",
    "bioterrorism", "biological weapon", "nerve agent", "blister agent",
    "blood agent", "choking agent", "incapacitating agent", "riot control agent",
    "toxic industrial chemicals", "hazardous materials", "hazmat",
    "personal protective equipment", "ppe", "respirator", "self-contained breathing apparatus",
    "scba", "decontamination", "mass casualty incident", "mci",
    "emergency medical services", "ems", "paramedic", "emergency room", "er",
    "intensive care unit", "icu", "ventilator", "dialysis", "transfusion",
    "surgery", "amputation", "burn unit", "rehabilitation", "physical therapy",
    "occupational therapy", "speech therapy", "prosthetics", "orthotics",
    "mental health services", "psychiatric care", "medication", "therapy",
    "support groups", "crisis hotline", "suicide prevention", "addiction treatment",
    "rehabilitation center", "sober living", "detoxification", "harm reduction",
    "needle exchange", "safe injection site", "overdose prevention", "naloxone",
    "emergency shelter", "temporary housing", "transitional housing",
    "permanent supportive housing", "homeless shelter", "soup kitchen",
    "food bank", "food pantry", "meal delivery", "clothing donation",
    "furniture donation", "household goods", "personal hygiene products",
    "diapers", "formula", "baby supplies", "school supplies", "books", "toys",
    "recreational activities", "arts and crafts", "sports", "music", "dance",
    "theater", "film", "creative writing", "storytelling", "poetry", "reading",
    "literacy", "education", "job training", "vocational skills",
    "entrepreneurship", "small business development", "financial literacy",
    "credit counseling", "debt management", "legal aid", "immigration services",
    "language interpretation", "translation services", "cultural sensitivity",
    "diversity training", "inclusion initiatives", "equity programs",
    "affirmative action", "civil rights", "humanitarian law", "international treaties",
    "diplomacy", "negotiation", "mediation", "conflict resolution",
    "peacebuilding", "reconciliation", "transitional justice", "reparations",
    "memorialization", "remembrance", "commemoration", "heritage preservation",
    "cultural revitalization", "traditional knowledge", "indigenous rights",
    "environmental justice", "sustainable agriculture", "renewable energy",
    "energy efficiency", "green building", "waste reduction", "recycling",
    "composting", "water conservation", "air purification", "soil remediation",
    "ecological restoration", "biodiversity conservation", "wildlife protection",
    "animal welfare", "vegetarianism", "veganism", "plant-based diet",
    "healthy eating", "nutrition", "exercise", "fitness", "wellness",
    "stress management", "mindfulness", "meditation", "yoga", "tai chi",
    "massage therapy", "acupuncture", "alternative medicine", "holistic health",
    "spiritual healing", "faith-based initiatives", "community organizing",
    "social activism", "political advocacy", "policy reform", "legislation",
    "government accountability", "citizen engagement", "civic participation",
    "democratic governance", "rule of law", "human rights advocacy",
    "peace and justice", "social change", "global citizenship",
    "sustainable living", "environmental stewardship", "planetary health",
    "human security", "well-being for all"
]

class DisasterAnalyzer:
    def __init__(self):
        self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
        self.model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=2)
        self.translator = Translator()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)

    def preprocess_text(self, text):
        """Clean and preprocess the tweet text using text_processing.py"""
        text = re.sub(r"http\S+", "", text)  # Remove URLs
        text = re.sub(r"@\w+", "", text)  # Remove mentions
        text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
        return text.lower()

    def extract_keywords(self, tweet):
        """Extract keywords from a tweet"""
        tokens = word_tokenize(self.preprocess_text(tweet))
        stop_words = set(stopwords.words("english"))
        return [word for word in tokens if word not in stop_words and word.isalpha()]

    def translate_keywords(self, keywords):
        """Translate keywords into English using Google Translate"""
        translated = []
        for keyword in keywords:
            try:
                translated.append(self.translator.translate(keyword, dest='en').text)
            except Exception as e:
                print(f"Translation error: {e}")
                translated.append(keyword)
        return translated

    def classify_with_xlm_roberta(self, tweet):
        """Classify tweet using XLM-RoBERTa model"""
        self.model.eval()
        inputs = self.tokenizer(tweet, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probabilities = F.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        return predicted_class

    def classify_by_keywords(self, translated_keywords):
        """Keyword-based classification"""
        return 1 if any(kw in DISASTER_KEYWORDS for kw in translated_keywords) else 0

    def prepare_data(self, file_path):
        """Load, clean, and prepare data for processing."""
        raw_data = load_data(file_path)
        cleaned_data = clean_data(raw_data)
        processed_data = process_text(cleaned_data)
        return processed_data

    def analyze_dataset(self, file_path):
        """Full analysis pipeline using integrated modules"""
        # Prepare data
        processed_data = self.prepare_data(file_path)

        # Visualization
        visualize_data(processed_data)

        # Sample analysis
        random_idx = random.randint(0, len(processed_data) - 1)
        sample_tweet = processed_data.iloc[random_idx]['Tweet Text']

        print("\nRandom Tweet Analysis:")
        print(f"Original Tweet: {sample_tweet}")
        print(f"Cleaned Text: {self.preprocess_text(sample_tweet)}")

        # Classify with XLM-RoBERTa model
        xlm_class_idx = self.classify_with_xlm_roberta(sample_tweet)
        classification_semantic = "Disaster" if xlm_class_idx == 1 else "Non-Disaster"

        # Classify by keywords
        keywords = self.translate_keywords(self.extract_keywords(sample_tweet))
        classification_keyword = "Disaster" if self.classify_by_keywords(keywords) == 1 else "Non-Disaster"

        print(f"Translated Keywords: {keywords}")
        print(f"Classification (Semantic): {classification_semantic}")
        print(f"Classification (Keyword): {classification_keyword}")

if __name__ == "__main__":
    analyzer = DisasterAnalyzer()
    file_path = r"C:\Users\abhis\Desktop\DisasterAnalysis\final_project2025\data\CLEANED_NEW_FINALDATASET_modifiednew.csv"
    analyzer.analyze_dataset(file_path)
