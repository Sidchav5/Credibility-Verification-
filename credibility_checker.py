import requests
import nltk
import numpy as np
import json
from newspaper import Article
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from googleapiclient.discovery import build
from transformers import pipeline

# Download necessary NLP resources
nltk.download('punkt')
nltk.download('stopwords')

# Load API keys from a config file
with open('config.json') as f:
    config = json.load(f)

SEARCH_API_KEY = config["SEARCH_API_KEY"]
FACT_CHECK_API_KEY = config["FACT_CHECK_API_KEY"]
CX = config["CX"]

# Summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)  # Use CPU

def extract_article_content(url):
    """Extract article content from a given URL."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        return None

def summarize_text(text):
    """Summarize the given article text."""
    if len(text) < 100:
        return text  # No need to summarize very short texts

    max_length = min(150, len(text) // 2)
    summary = summarizer(text, max_length=max_length, min_length=50, do_sample=False)
    return summary[0]['summary_text']

def extract_keywords(text):
    """Extract keywords from the given text."""
    stop_words = set(nltk.corpus.stopwords.words('english'))
    words = nltk.word_tokenize(text)
    return [word for word in words if word.isalnum() and word.lower() not in stop_words]

def search_related_articles(keywords):
    """Search related articles using Google Custom Search API."""
    query = '+'.join(keywords)
    service = build("customsearch", "v1", developerKey=SEARCH_API_KEY)
    
    try:
        res = service.cse().list(q=query, cx=CX).execute()
        results = res.get('items', [])
        return [(item['title'], item['link']) for item in results[:5]]  # Return top 5 articles
    except Exception as e:
        return []

def fetch_and_summarize_article(url):
    """Fetch and summarize an article from a given URL."""
    text = extract_article_content(url)
    return summarize_text(text) if text else None

def compare_summaries(original_summary, related_summaries):
    """Compute cosine similarity scores between the original summary and related articles."""
    vectorizer = TfidfVectorizer().fit_transform([original_summary] + related_summaries)
    return cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()

def classify_article(similarity_scores):
    """Classify the credibility of an article based on similarity scores."""
    avg_score = np.mean(similarity_scores) if similarity_scores.size > 0 else 0

    if avg_score < 0.05:
        return "Fake"
    elif avg_score < 0.25:
        return "Suspicious"
    else:
        return "Correct"

def get_fact_check_results(keywords):
    """Query Google Fact Check API for verification results."""
    query = ' '.join(keywords)
    service = build("factchecktools", "v1alpha1", developerKey=FACT_CHECK_API_KEY)

    try:
        res = service.claims().search(query=query).execute()
        return [
            {"claim": item['claimReview'][0].get('textualBody', 'No claim text available'), "url": item['claimReview'][0].get('url', 'No URL available')}
            for item in res.get('claims', [])
        ]
    except Exception:
        return []
