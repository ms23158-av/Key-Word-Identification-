

import requests
from bs4 import BeautifulSoup
import 
import sqlite3
from datetime import datetime
from urllib.parse import urljoin, urlparse
import pandas as pd
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import networkx as nx
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

MAX_PAGES = 15          
DB_PATH = "keyword_project_enhanced.db"
SITES = {
    "IIM_Ahmedabad": "https://www.iima.ac.in/",
    "IISER_Mohali": "https://www.iisermohali.ac.in/"
}

def create_db(conn):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS pages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            site TEXT,
            url TEXT,
            status INTEGER,
            text_blob TEXT,
            title TEXT,
            scraped_at TEXT
        )
    """)
    conn.commit()

def store_page(conn, site, url, status, text_blob, title):
    cur = conn.cursor()
    scraped_at = datetime.utcnow().isoformat()
    cur.execute("""
        INSERT INTO pages (site,url,status,text_blob,title,scraped_at)
        VALUES (?,?,?,?,?,?)
    """, (site, url, status, text_blob, title, scraped_at))
    conn.commit()

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text.lower().strip()

def extract_visible_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(['script', 'style', 'noscript']):
        tag.decompose()
    return soup.get_text(separator=' '), soup.title.string if soup.title else ""

def crawl_site(site_name, base_url, conn):
    visited, to_visit = set(), [base_url]
    count = 0

    while to_visit and count < MAX_PAGES:
        url = to_visit.pop(0)
        if url in visited:
            continue

        try:
            r = requests.get(url, timeout=10)
            visited.add(url)
            text, title = extract_visible_text(r.text)
            clean = clean_text(text)
            store_page(conn, site_name, url, r.status_code, clean, title)
            count += 1
            print(f"[{site_name}] Page {count} stored: {url}")

            # collect internal links
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=True):
                full = urljoin(url, a["href"])
                if base_url in full and full not in visited and len(to_visit) < 100:
                    to_visit.append(full)

        except Exception as e:
            print(f"Failed: {url} ({e})")

def preprocess_texts(texts):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    cleaned_texts = []
    for text in texts:
        tokens = nltk.word_tokenize(text)
        tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalpha() and t not in stop_words]
        cleaned_texts.append(' '.join(tokens))
    return cleaned_texts

def analyze_keywords(conn):
    df = pd.read_sql_query("SELECT site, text_blob FROM pages", conn)
    df['clean_text'] = preprocess_texts(df['text_blob'])
    
    
    all_counts = {}
    for site, group in df.groupby('site'):
        words = ' '.join(group['clean_text']).split()
        counter = Counter(words)
        all_counts[site] = counter.most_common(20)

    return df, all_counts

def plot_bar_charts(all_counts):
    for site, words in all_counts.items():
        data = pd.DataFrame(words, columns=['word', 'count'])
        plt.figure(figsize=(8,5))
        sns.barplot(data=data, x='count', y='word', palette='Blues_d')
        plt.title(f"Top Words - {site}")
        plt.tight_layout()
        plt.savefig(f"{site}_bar_chart.png")
        plt.show()

def plot_word_cloud(df):
    for site, group in df.groupby('site'):
        text = ' '.join(group['clean_text'])
        wc = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(8,4))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.title(f"Word Cloud - {site}")
        plt.savefig(f"{site}_wordcloud.png")
        plt.show()

def plot_tfidf_heatmap(df):
    vectorizer = TfidfVectorizer(max_features=20)
    tfidf = vectorizer.fit_transform(df['clean_text'])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_df = pd.DataFrame(tfidf.toarray(), columns=feature_names)
    tfidf_df['site'] = df['site']
    avg_tfidf = tfidf_df.groupby('site').mean()

    plt.figure(figsize=(10,6))
    sns.heatmap(avg_tfidf, cmap="YlGnBu", annot=True)
    plt.title("TF-IDF Heatmap (Top 20 Features)")
    plt.tight_layout()
    plt.savefig("tfidf_heatmap.png")
    plt.show()

def plot_cooccurrence_network(df):
    for site, group in df.groupby('site'):
        words = ' '.join(group['clean_text']).split()
        pairs = defaultdict(int)
        for i in range(len(words) - 1):
            pairs[(words[i], words[i+1])] += 1

        G = nx.Graph()
        for (w1, w2), freq in pairs.items():
            if freq > 2:
                G.add_edge(w1, w2, weight=freq)

        plt.figure(figsize=(8,6))
        pos = nx.spring_layout(G, k=0.5)
        nx.draw_networkx(G, pos, with_labels=True, node_size=50, font_size=8)
        plt.title(f"Word Co-occurrence Network - {site}")
        plt.tight_layout()
        plt.savefig(f"{site}_network.png")
        plt.show()

def main():
    conn = sqlite3.connect(DB_PATH)
    create_db(conn)

    # Crawl sites
    for site_name, url in SITES.items():
        crawl_site(site_name, url, conn)

    # Analyze & visualize
    df, all_counts = analyze_keywords(conn)
    plot_bar_charts(all_counts)
    plot_word_cloud(df)
    plot_tfidf_heatmap(df)
    plot_cooccurrence_network(df)
    conn.close()

if __name__ == "__main__":
    main()
