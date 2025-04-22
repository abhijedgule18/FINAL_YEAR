import streamlit as st
import pandas as pd
import random
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Import your project modules
# Conditional imports to avoid the torch error in Streamlit's file watcher
if 'torch' not in st.session_state:
    import torch
    import torch.nn.functional as F
    from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
    from googletrans import Translator

    st.session_state['torch'] = True

# Import your project modules - use try/except to handle possible import errors
try:
    from src.data_cleaning import load_data, clean_data
    from src.text_processing import process_text
    from src.visualization import visualize_data, visualize_model_results
    from src import model_training
except ImportError:
    st.error(
        "Could not import required modules from 'src'. Make sure the application is run from the correct directory.")
    st.stop()


# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)


# Define disaster keywords
DISASTER_KEYWORDS = [ "earthquake", "flood", "hurricane", "typhoon", "fire", "landslide",
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
    "human security", "well-being for all]


class DisasterAnalyzer:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.translator = None
        self.device = None

    def initialize_models(self):
        progress_text = st.empty()
        progress_text.text("Loading models...")

        # Import torch modules here instead of at the top level
        import torch
        import torch.nn.functional as F
        from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
        from googletrans import Translator

        self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
        self.model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=2)
        self.translator = Translator()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)
        progress_text.text("Models loaded successfully!")

    def preprocess_text(self, text):
        text = re.sub(r"http\S+", "", str(text))
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"[^\w\s]", "", text)
        return text.lower()

    def extract_keywords(self, tweet):
        tokens = word_tokenize(self.preprocess_text(tweet))
        stop_words = set(stopwords.words("english"))
        return [word for word in tokens if word not in stop_words and word.isalpha()]

    def translate_keywords(self, keywords):
        translated = []
        for keyword in keywords:
            try:
                translated.append(self.translator.translate(keyword, dest='en').text)
            except Exception as e:

                translated.append(keyword)
        return translated

    def classify_with_xlm_roberta(self, tweet):
        # Import torch modules here
        import torch
        import torch.nn.functional as F

        self.model.eval()
        inputs = self.tokenizer(tweet, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probabilities = F.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        return predicted_class, confidence

    def classify_by_keywords(self, translated_keywords):
        matches = [kw for kw in translated_keywords if kw in DISASTER_KEYWORDS]
        return 1 if matches else 0, matches

    def prepare_data(self, data_frame):
        progress_text = st.empty()
        progress_text.text("Cleaning data...")

        # If data_frame came from a file upload, it's already a dataframe
        # Otherwise, assume it came from load_data
        if isinstance(data_frame, pd.DataFrame):
            cleaned_data = clean_data(data_frame)
        else:
            # Handle the case where data_frame might be a path
            raw_data = load_data(data_frame) if isinstance(data_frame, str) else data_frame
            cleaned_data = clean_data(raw_data)

        progress_text.text("Processing text...")
        processed_data = process_text(cleaned_data)
        progress_text.empty()
        return processed_data

    def analyze_dataset(self, data_frame):
        # Initialize models if not already done
        if self.model is None:
            self.initialize_models()

        progress_bar = st.progress(0)
        progress_text = st.empty()

        # Prepare data
        progress_text.text("Preparing data...")
        progress_bar.progress(10)
        processed_data = self.prepare_data(data_frame)

        # Create results container
        results_container = st.container()

        with results_container:
            st.subheader("Analysis Results")

            # Display data summary
            progress_text.text("Analyzing dataset...")
            progress_bar.progress(30)

            # Calculate dataset statistics
            total_tweets = len(processed_data)

            # Check if 'is_disaster' column exists for classification stats
            disaster_stats = {}
            if 'is_disaster' in processed_data.columns:
                processed_data['is_disaster'] = processed_data['is_disaster'].astype(int)
                disaster_count = processed_data['is_disaster'].sum()
                non_disaster_count = total_tweets - disaster_count
                disaster_percent = (disaster_count / total_tweets) * 100 if total_tweets > 0 else 0

                disaster_stats = {
                    "Total Tweets": total_tweets,
                    "Disaster Tweets": disaster_count,
                    "Non-Disaster Tweets": non_disaster_count,
                    "Disaster Percentage": f"{disaster_percent:.1f}%"
                }
            else:
                disaster_stats = {
                    "Total Tweets": total_tweets,
                    "Classification": "No pre-labeled data available"
                }

            # Display dataset statistics in a table
            st.subheader("Dataset Summary")
            stats_df = pd.DataFrame([disaster_stats])
            st.table(stats_df.T.rename(columns={0: "Value"}))

            # Analyze random tweets
            st.subheader("Random Tweet Analysis")
            progress_bar.progress(50)

            if total_tweets > 0:
                # Select multiple random tweets
                num_samples = min(3, total_tweets)  # Analyze up to 3 random tweets
                random_indices = random.sample(range(total_tweets), num_samples)

                # Create lists to store results
                original_tweets = []
                cleaned_texts = []
                keywords_list = []
                semantic_classifications = []
                semantic_confidences = []
                keyword_classifications = []
                keyword_matches = []

                # Analyze each tweet
                for idx in random_indices:
                    sample_tweet = processed_data.iloc[idx]['Tweet Text']
                    cleaned_text = self.preprocess_text(sample_tweet)
                    keywords = self.extract_keywords(sample_tweet)
                    translated_kw = self.translate_keywords(keywords)

                    xlm_class_idx, confidence = self.classify_with_xlm_roberta(sample_tweet)
                    classification_semantic = "Disaster" if xlm_class_idx == 1 else "Non-Disaster"

                    keyword_class, matching_keywords = self.classify_by_keywords(translated_kw)
                    classification_keyword = "Disaster" if keyword_class == 1 else "Non-Disaster"

                    # Append results to lists
                    original_tweets.append(sample_tweet)
                    cleaned_texts.append(cleaned_text)
                    keywords_list.append(", ".join(translated_kw))
                    semantic_classifications.append(classification_semantic)
                    semantic_confidences.append(f"{confidence:.2%}")
                    keyword_classifications.append(classification_keyword)
                    keyword_matches.append(", ".join(matching_keywords) if matching_keywords else "None")

                # Create results DataFrame
                tweet_results = pd.DataFrame({
                    "Original Tweet": original_tweets,
                    "Cleaned Text": cleaned_texts,
                    "Keywords": keywords_list,
                    "Semantic Classification": semantic_classifications,
                    "Confidence": semantic_confidences,
                    "Keyword Classification": keyword_classifications,
                    "Matching Disaster Keywords": keyword_matches
                })

                # Display results in a table
                st.table(tweet_results)
            else:
                st.error("No data available for analysis. Please check your dataset.")

        # Train and evaluate models
        st.subheader("Model Training & Evaluation")
        progress_text.text("Training models...")
        progress_bar.progress(80)

        try:
            model_results = model_training.train_model(processed_data)

            # Process model results for display
            if isinstance(model_results, tuple) and len(model_results) == 5:
                xgb_model, lgb_model, vectorizer, label_encoder, results = model_results
            else:
                results = model_results

            # Format results for display in a table
            model_performance = self.format_model_results(results)

            # Display model performance table
            st.subheader("Model Performance")
            st.table(model_performance)

            progress_text.text("Analysis completed!")
            progress_bar.progress(100)
        except Exception as e:
            st.error(f"Error during model training: {str(e)}")
            import traceback
            st.code(traceback.format_exc(), language="python")
            progress_bar.progress(100)

    def format_model_results(self, results):
        # Handle different formats of results and convert to a DataFrame for display
        try:
            # Case 1: results is a dictionary with model names as keys
            if isinstance(results, dict) and hasattr(results, 'keys'):
                model_names = list(results.keys())
                metrics = pd.DataFrame()

                for metric in ['accuracy', 'precision', 'recall', 'f1']:
                    if all(metric in results[model] for model in model_names):
                        metrics[metric.capitalize()] = [results[model][metric] for model in model_names]

                metrics.index = model_names
                return metrics

            # Case 2: results is a list of dictionaries with 'model' and metrics
            elif isinstance(results, list) and all(isinstance(item, dict) for item in results):
                df = pd.DataFrame(results)

                if 'model' in df.columns:
                    df = df.set_index('model')
                    print(df)
                return df

            # Case 3: results is a list of accuracies
            elif isinstance(results, list) and all(isinstance(item, (int, float)) for item in results):
                return pd.DataFrame({
                    'Accuracy': results,
                    'Model': [f"Model {i + 1}" for i in range(len(results))]
                }).set_index('Model')

            # Case 4: results might be a single model with metrics as attributes
            elif hasattr(results, 'accuracy'):
                metrics = {}
                for metric in ['accuracy', 'precision', 'recall', 'f1']:
                    if hasattr(results, metric):
                        metrics[metric.capitalize()] = getattr(results, metric)
                return pd.DataFrame([metrics], index=['Model'])

            else:
                # Default case if structure is unknown
                return pd.DataFrame([{'Result': f"Unknown format: {type(results)}"}])

        except Exception as e:
            return pd.DataFrame([{'Error': f"Could not format results: {str(e)}"}])


def main():
    st.title("Disaster Tweet Analyzer")
    st.write("""
    Upload a CSV file containing tweets to analyze whether they are related to disasters.
    The app will clean the data, process the text, and classify tweets using both semantic and keyword-based approaches.
    """)

    # Download NLTK data at startup
    download_nltk_data()

    analyzer = DisasterAnalyzer()

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        st.success("File uploaded successfully!")

        try:
            # Preview the data
            df = pd.read_csv(uploaded_file)
            st.subheader("Data Preview")
            st.dataframe(df.head())

            # Check if required columns are present or can be created
            if 'Tweet Text' not in df.columns:
                # Try to find a suitable text column
                text_columns = [col for col in df.columns if 'text' in col.lower() or 'tweet' in col.lower()]

                if text_columns:
                    # Use the first matching column
                    df['Tweet Text'] = df[text_columns[0]]
                    st.info(f"Using '{text_columns[0]}' as the 'Tweet Text' column.")
                else:
                    st.warning(
                        "The uploaded file does not have a 'Tweet Text' column. Please ensure your data has the correct format.")
                    st.stop()

            # Add start analysis button
            if st.button("Start Analysis"):
                with st.spinner("Analyzing data..."):
                    analyzer.analyze_dataset(df)

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            import traceback
            st.code(traceback.format_exc(), language="python")

    st.sidebar.title("About")
    st.sidebar.info("""
    This application analyzes tweets to determine if they are related to disasters.
    It uses both semantic analysis with XLM-RoBERTa and keyword-based analysis.
    """)

    # Add option to analyze a single tweet
    st.sidebar.subheader("Quick Tweet Analysis")
    sample_tweet = st.sidebar.text_area("Enter a tweet to analyze:")

    if sample_tweet and st.sidebar.button("Analyze Tweet"):
        # Initialize models if not already done
        if analyzer.model is None:
            analyzer.initialize_models()

        with st.sidebar:
            with st.spinner("Analyzing tweet..."):
                cleaned_text = analyzer.preprocess_text(sample_tweet)
                keywords = analyzer.extract_keywords(sample_tweet)
                translated_kw = analyzer.translate_keywords(keywords)

                xlm_class_idx, confidence = analyzer.classify_with_xlm_roberta(sample_tweet)
                classification = "Disaster" if xlm_class_idx == 1 else "Non-Disaster"

                keyword_class, matches = analyzer.classify_by_keywords(translated_kw)
                kw_classification = "Disaster" if keyword_class == 1 else "Non-Disaster"

                # Display results in a more tabular format
                results_df = pd.DataFrame({
                    "Analysis": ["Cleaned Text", "Keywords", "Semantic Classification", "Confidence",
                                 "Keyword Classification", "Matching Keywords"],
                    "Result": [
                        cleaned_text,
                        ", ".join(translated_kw),
                        classification,
                        f"{confidence:.2%}",
                        kw_classification,
                        ", ".join(matches) if matches else "None"
                    ]
                }).set_index("Analysis")

                st.table(results_df)


if __name__ == "__main__":
    main()
