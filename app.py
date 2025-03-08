from flask import Flask, render_template, request, jsonify, send_file
import os
import nltk
import numpy as np
import matplotlib.pyplot as plt
from newspaper import Article
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from nltk.tokenize import sent_tokenize
from googleapiclient.discovery import build

# Set Matplotlib backend to 'Agg' to avoid GUI warnings
import matplotlib
matplotlib.use('Agg')

# Download necessary NLP resources
nltk.download('punkt')

app = Flask(__name__)

# API Keys and Search Engine ID (Replace with your actual keys)
SEARCH_API_KEY = ""
FACT_CHECK_API_KEY = ""
CX = ""

# Ensure the static folder exists for storing graphs
GRAPH_FOLDER = "static/graphs"
os.makedirs(GRAPH_FOLDER, exist_ok=True)

# Load NLI pipeline
nli_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Load Sentence Transformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# 📌 Function to extract article content
from sklearn.feature_extraction.text import TfidfVectorizer
from newspaper import Article

# 📌 Function to extract article content and headline keywords
def extract_article_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        headline = article.title
        # Extract keywords from the headline using TF-IDF vectorizer
        keywords = extract_keywords(headline)
        return article.text, keywords
    except Exception as e:
        print(f"Error extracting article: {str(e)}")
        return None, []


# 📌 Function to summarize text
def summarize_text(text, num_sentences=5):
    if not text.strip():
        return "No content available for summarization."
    sentences = sent_tokenize(text)
    return " ".join(sentences[:num_sentences]) if len(sentences) > num_sentences else " ".join(sentences)

# 📌 Function to check contradiction using NLI model
def check_contradiction(text1, text2):
    result = nli_model(text1, candidate_labels=["entailment", "contradiction", "neutral"])
    return result["labels"][1] == "contradiction"

# 📌 Function to extract keywords from the summary
def extract_keywords(summary, num_keywords=5):
    if not summary.strip():
        return []
    
    vectorizer = TfidfVectorizer(stop_words="english", max_features=num_keywords)
    tfidf_matrix = vectorizer.fit_transform([summary])
    keywords = vectorizer.get_feature_names_out()
    
    return list(keywords)

# 📌 Function to search related articles using the extracted keywords
def search_related_articles(keywords):
    try:
        query = '+'.join(keywords)
        service = build("customsearch", "v1", developerKey=SEARCH_API_KEY)
        res = service.cse().list(q=query, cx=CX).execute()

        # Check if search results are available
        if 'items' in res:
            return [(item['title'], item['link']) for item in res['items'][:5]]  # Return top 5 results
        else:
            print("No search results found.")
            return []
    
    except Exception as e:
        print("Error in Google Search API:", str(e))
        return []

# 📌 Function to classify article credibility based on similarity scores
def classify_article(similarity_scores):
    if len(similarity_scores) == 0:
        return "No related articles found"
    
    avg_score = np.mean(similarity_scores)
    if avg_score < 0.05:
        return "Fake"
    elif avg_score < 0.25:
        return "Suspicious"
    else:
        return "Correct"

# 📌 Function to compare summaries using embeddings and calculate cosine similarity
def compare_summaries(original_summary, related_summaries):
    embeddings = embedding_model.encode([original_summary] + related_summaries)
    cosine_sim = util.pytorch_cos_sim(embeddings[0], embeddings[1:])
    return cosine_sim[0].tolist()

# 📌 Function to generate a similarity graph
def plot_similarity_graph(similarity_scores, article_titles):
    if len(similarity_scores) == 0:
        return None
    
    plt.figure(figsize=(8, 5))
    indices = [str(i + 1) for i in range(len(similarity_scores))]
    plt.bar(indices, similarity_scores, color='skyblue')
    plt.xlabel('Related Articles (Index)')
    plt.ylabel('Cosine Similarity')
    plt.title('Cosine Similarity with Related Articles')
    plt.xticks(indices)
    graph_path = os.path.join(GRAPH_FOLDER, "similarity_graph.png")
    plt.savefig(graph_path)
    plt.close()
    return graph_path

# 📌 Function to get fact-check results
def get_fact_check_results(keywords):
    try:
        query = ' '.join(keywords)
        service = build("factchecktools", "v1alpha1", developerKey=FACT_CHECK_API_KEY)
        res = service.claims().search(query=query).execute()

        return [
            {'claim': item['claimReview'][0].get('textualBody', ''), 'url': item['claimReview'][0].get('url', '')}
            for item in res.get('claims', []) if 'claimReview' in item and item['claimReview']
        ]
    
    except Exception as e:
        print("Error in fact check API:", str(e))
        return []

# 📌 Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    input_type = data.get('inputType')
    content = data.get('content')

    if input_type == "url":
        article_content, headline_keywords = extract_article_content(content)
        summary = summarize_text(article_content) if article_content else "Error retrieving article content."
    else:
        summary = content
        headline_keywords = extract_keywords(summary)

    return jsonify({
        "summary": summary,
        "keywords": headline_keywords  # Send extracted keywords as part of the response
    })


@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.json
        keywords = data.get('keywords')
        original_summary = data.get('summary')

        # Automatically extract keywords if not provided
        if not keywords or len(keywords) == 0:
            keywords = extract_keywords(original_summary)
            print("Auto-extracted keywords:", keywords)

        # Search related articles
        related_articles = search_related_articles(keywords)
        if not related_articles:
            return jsonify({"error": "No related articles found. Try different keywords."}), 400

        # Summarize related articles
        related_summaries = [
            summarize_text(extract_article_content(url)[0]) for _, url in related_articles
        ]
        print("Summarized Articles:", related_summaries)

        # Compare summaries and calculate similarity scores
        similarity_scores = compare_summaries(original_summary, related_summaries)
        print("Similarity Scores:", similarity_scores)

        # Classify article based on similarity scores
        classification = classify_article(similarity_scores)
        print("Article Classification:", classification)

        # Check for contradictions
        contradictions = [
            "Contradiction detected" if check_contradiction(original_summary, summary) else "No contradiction"
            for summary in related_summaries
        ]

        # Get fact-check results
        fact_check_results = get_fact_check_results(keywords)

        # Generate similarity graph
        graph_path = plot_similarity_graph(similarity_scores, [title for title, _ in related_articles])
        if graph_path:
            print("Graph Generated at:", graph_path)

        # Return structured results
        return jsonify({
            "classification": classification,
            "avg_score": round(np.mean(similarity_scores), 2) if similarity_scores else None,
            "articles": [{"title": title, "link": link, "score": round(score, 2), "contradiction": contradiction}
                         for (title, link), score, contradiction in zip(related_articles, similarity_scores, contradictions)],
            "fact_check": fact_check_results,
            "graph_url": "/" + graph_path.replace("\\", "/") if graph_path else None
        })

    except Exception as e:
        print("Error in search route:", str(e))
        return jsonify({"error": "An error occurred while searching articles."}), 500



@app.route('/similarity_graph')
def get_similarity_graph():
    return send_file(os.path.join(GRAPH_FOLDER, "similarity_graph.png"), mimetype='image/png')


# 📌 Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
