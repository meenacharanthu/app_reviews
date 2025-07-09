# app.py (with BM25 search)
from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import pandas as pd
import sqlite3
import re
import numpy as np
import joblib
import os
from textblob import TextBlob
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['SUPERVISOR_CREDENTIALS'] = {'username': 'admin', 'password': 'admin123'}

# Load datasets
apps_df = pd.read_csv('data/googleplaystore.csv')
reviews_df = pd.read_csv('data/googleplaystore_user_reviews.csv')

# Preprocess data
apps_df = apps_df.drop_duplicates('App')
apps_df = apps_df.dropna(subset=['App'])
app_names = sorted(apps_df['App'].unique().tolist())

# Initialize database
def init_db():
    conn = sqlite3.connect('reviews.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS user_reviews
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 app_name TEXT NOT NULL,
                 review_text TEXT NOT NULL,
                 sentiment TEXT,
                 status TEXT DEFAULT 'pending',
                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

init_db()

# Initialize NLP resources
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Text preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords and lemmatize
    processed_tokens = []
    for token in tokens:
        if token not in stop_words and len(token) > 2:
            lemma = lemmatizer.lemmatize(token)
            processed_tokens.append(lemma)
    
    return " ".join(processed_tokens)

# Prepare search index
def create_search_index():
    print("Creating BM25 search index...")
    
    # Create document corpus for each app
    apps_df['search_text'] = (
        apps_df['App'] + ' ' + 
        apps_df['App'] + ' ' + 
        apps_df['Category'] + ' ' + 
        apps_df['Genres'].fillna('') + ' ' +
        apps_df['Content Rating'].fillna('')
    )
    
    # Preprocess text
    apps_df['processed_text'] = apps_df['search_text'].apply(preprocess_text)
    
    # Tokenize documents for BM25
    tokenized_corpus = [doc.split() for doc in apps_df['processed_text']]
    
    # Create BM25 index
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Save artifacts
    joblib.dump(bm25, 'bm25_index.joblib')
    apps_df.to_pickle('apps_df.pkl')
    print("BM25 index created successfully")

# Create search index if not exists
if not os.path.exists('bm25_index.joblib'):
    create_search_index()

# Load search artifacts
bm25 = joblib.load('bm25_index.joblib')
apps_df = pd.read_pickle('apps_df.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search')
def search_suggestions():
    query = request.args.get('q', '').lower()
    if len(query) >= 3:
        # Simple prefix matching for suggestions
        matches = [app for app in app_names if query in app.lower()]
        return jsonify(matches[:15])
    return jsonify([])




# @app.route('/search_results', methods=['GET'])
# def search_results():
#     query = request.args.get('q', '').strip()
#     if not query:
#         return redirect(url_for('index'))
    
#     # Preprocess query
#     processed_query = preprocess_text(query)
#     tokenized_query = processed_query.split()
    
#     # Get BM25 scores
#     scores = bm25.get_scores(tokenized_query)
    
#     # Get top 10 results
#     top_indices = np.argsort(scores)[::-1][:10]
#     results = []
#     for idx in top_indices:
#         app_data = apps_df.iloc[idx].to_dict()
#         # Use 'similarity' key instead of 'score'
#         # if scores[idx] > 0:
#         app_data['similarity'] = scores[idx]  # Changed from 'score' to 'similarity'
#         results.append(app_data)
    
#     return render_template('search_results.html', results=results, query=query)

@app.route('/search_results', methods=['GET'])
def search_results():
    query = request.args.get('q', '').strip()
    if not query:
        return redirect(url_for('index'))
    
    # Preprocess and tokenize query
    processed_query = preprocess_text(query)
    tokenized_query = processed_query.split()
    
    # Get BM25 scores
    scores = bm25.get_scores(tokenized_query)
    
    results = []
    seen = set()

    # Get top 10 by BM25
    top_indices = np.argsort(scores)[::-1][:10]
    for idx in top_indices:
        if scores[idx] > 0:
            app_data = apps_df.iloc[idx].to_dict()
            app_data['similarity'] = scores[idx]
            results.append(app_data)
            seen.add(app_data['App'])

    # Fallback: Add prefix matches with score = 0.01 (if not already in top BM25)
    if len(results) < 10 and len(query) <= 4:
        extra_matches = [app for app in app_names if query.lower() in app.lower()]
        for match in extra_matches:
            if match not in seen:
                row = apps_df[apps_df['App'] == match].iloc[0].to_dict()
                row['similarity'] = 0.01  # Low fallback score
                results.append(row)
                seen.add(match)
            if len(results) >= 10:
                break

    return render_template('search_results.html', results=results, query=query)


@app.route('/app/<app_name>')
def app_details(app_name):
    # Get app details
    app_data = apps_df[apps_df['App'] == app_name].iloc[0].to_dict()
    
    # Get existing reviews from CSV
    csv_reviews = reviews_df[reviews_df['App'] == app_name].to_dict('records')
    
    # Get approved user reviews from DB
    conn = sqlite3.connect('reviews.db')
    c = conn.cursor()
    c.execute("SELECT * FROM user_reviews WHERE app_name = ? AND status = 'approved'", (app_name,))
    db_reviews = [dict(id=row[0], app_name=row[1], review_text=row[2], sentiment=row[3]) 
                 for row in c.fetchall()]
    conn.close()
    
    all_reviews = csv_reviews + db_reviews
    return render_template('app_detail.html', app=app_data, reviews=all_reviews)

@app.route('/submit_review', methods=['POST'])
def submit_review():
    app_name = request.form['app_name']
    review_text = request.form['review_text']
    
    # Analyze sentiment
    sentiment = analyze_sentiment(review_text)
    
    conn = sqlite3.connect('reviews.db')
    c = conn.cursor()
    c.execute("INSERT INTO user_reviews (app_name, review_text, sentiment) VALUES (?, ?, ?)",
              (app_name, review_text, sentiment))
    conn.commit()
    conn.close()
    
    return redirect(url_for('app_details', app_name=app_name))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if (username == app.config['SUPERVISOR_CREDENTIALS']['username'] and 
            password == app.config['SUPERVISOR_CREDENTIALS']['password']):
            session['supervisor'] = True
            return redirect(url_for('approvals'))
        return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@app.route('/approvals')
def approvals():
    if not session.get('supervisor'):
        return redirect(url_for('login'))
    
    conn = sqlite3.connect('reviews.db')
    c = conn.cursor()
    c.execute("SELECT * FROM user_reviews WHERE status = 'pending'")
    reviews = [dict(id=row[0], app_name=row[1], review_text=row[2], sentiment=row[3], created_at=row[5]) 
              for row in c.fetchall()]
    conn.close()
    
    return render_template('approvals.html', reviews=reviews)

@app.route('/approve_review/<int:review_id>')
def approve_review(review_id):
    if not session.get('supervisor'):
        return redirect(url_for('login'))
    
    conn = sqlite3.connect('reviews.db')
    c = conn.cursor()
    c.execute("UPDATE user_reviews SET status = 'approved' WHERE id = ?", (review_id,))
    conn.commit()
    conn.close()
    return redirect(url_for('approvals'))

@app.route('/reject_review/<int:review_id>')
def reject_review(review_id):
    if not session.get('supervisor'):
        return redirect(url_for('login'))
    
    conn = sqlite3.connect('reviews.db')
    c = conn.cursor()
    c.execute("UPDATE user_reviews SET status = 'rejected' WHERE id = ?", (review_id,))
    conn.commit()
    conn.close()
    return redirect(url_for('approvals'))

def analyze_sentiment(text):
    analysis = TextBlob(text)
    # Convert polarity to a sentiment category
    if analysis.sentiment.polarity > 0.2:
        return 'Positive'
    elif analysis.sentiment.polarity < -0.2:
        return 'Negative'
    else:
        return 'Neutral'

if __name__ == '__main__':
    app.run(debug=True) 