
import time
import sqlite3
import requests
from urllib.parse import urljoin, urlparse
from urllib import robotparser
from bs4 import BeautifulSoup
from tqdm import tqdm
import re
from collections import Counter
import csv
import os
from datetime import datetime

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

TARGETS = [
    ("IIMA", "https://www.iima.ac.in/"),
    ("IISER_MOHALI", "https://www.iisermohali.ac.in/"),
    # Add more specific subpaths if you like, e.g.:
    # ("IIMA_news", "https://www.iima.ac.in/news"),
    # ("IISER_faculty", "https://www.iisermohali.ac.in/faculty"),
]

USER_AGENT = "KeywordScraper/1.0 (email: your-email@example.com)"
REQUEST_DELAY_SECONDS = 1.0   
TIMEOUT = 10  

DB_PATH = "keyword_project_scrape.db"
CSV_TOP_WORDS = "keyword_top_words.csv"

STOPWORDS = {
    'the','and','a','to','of','in','is','that','this','as','for','with','on','by','be','are',
    'was','it','an','at','from','we','our','has','have','their','which','or','into','such','these',
    'its','will','not','also','but','they','about','more','per','since','than','been','page','home',
    'contact','read','download','menu','site'
}


def can_fetch(url, agent=USER_AGENT):
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    rp = robotparser.RobotFileParser()
    rp.set_url(urljoin(base, "/robots.txt"))
    try:
        rp.read()
    except Exception:
        
        return True
    return rp.can_fetch(agent, url)

def fetch_url(url):
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, headers=headers, timeout=TIMEOUT)
    r.raise_for_status()
    return r.text

def extract_text(html):
    
    soup = BeautifulSoup(html, "lxml")
    for s in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        s.extract()
   
    main = soup.find("main")
    if main:
        text = main.get_text(separator=" ", strip=True)
    else:
        text = soup.get_text(separator=" ", strip=True)
    
    text = re.sub(r'\s+', ' ', text)
    return text

def init_db(path=DB_PATH):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS pages (
                    id INTEGER PRIMARY KEY,
                    site TEXT,
                    url TEXT,
                    status INTEGER,
                    text_blob TEXT,
                    title TEXT,
                    scraped_at TEXT
                )""")
    conn.commit()
    return conn

def store_page(conn, site, url, status, text_blob, title):
    cur = conn.cursor()
    scraped_at = datetime.utcnow().isoformat()
    cur.execute("INSERT INTO pages (site,url,status,text_blob,title,scraped_at) VALUES (?,?,?,?,?)",
                (site, url, status, text_blob, title, scraped_at))
    conn.commit()

TOKEN_RE = re.compile(r"[A-Za-z']+")

def tokenize(text):
    tokens = TOKEN_RE.findall(text.lower())
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens

def top_k_words(text, k=30):
    tokens = tokenize(text)
    c = Counter(tokens)
    return c.most_common(k)


def compute_tfidf_top_terms(docs, top_n=10):
    if not SKLEARN_AVAILABLE:
        print("sklearn not available: skipping TF-IDF")
        return {}
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(docs)
    feature_names = vectorizer.get_feature_names_out()
    results = {}
    for i, row in enumerate(X):
        rowdata = row.toarray().ravel()
        top_idx = rowdata.argsort()[-top_n:][::-1]
        top_terms = [(feature_names[idx], float(rowdata[idx])) for idx in top_idx if rowdata[idx] > 0]
        results[i] = top_terms
    return results

def main():
    conn = init_db()
    saved_texts = []
    saved_meta = []

    for site_name, start_url in tqdm(TARGETS, desc="targets"):
        if not can_fetch(start_url):
            print(f"[WARN] robots.txt disallows fetching {start_url}. Skipping.")
            continue

        try:
            html = fetch_url(start_url)
        except Exception as e:
            print(f"[ERROR] fetching {start_url}: {e}")
            store_page(conn, site_name, start_url, 0, "", "")
            continue

        text = extract_text(html)
       
        soup = BeautifulSoup(html, "lxml")
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""

        store_page(conn, site_name, start_url, 200, text, title)
        saved_texts.append(text)
        saved_meta.append((site_name, start_url, title))
        time.sleep(REQUEST_DELAY_SECONDS)

        
    rows = []
    for i, (site_name, url, title) in enumerate(saved_meta):
        text = saved_texts[i]
        top = top_k_words(text, k=50)
        for w, cnt in top:
            rows.append({'site': site_name, 'url': url, 'title': title, 'word': w, 'count': cnt})

    
    with open(CSV_TOP_WORDS, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['site','url','title','word','count'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Saved top-word summary to {CSV_TOP_WORDS}")

    
    if saved_texts and SKLEARN_AVAILABLE:
        tfidf_res = compute_tfidf_top_terms(saved_texts, top_n=15)
        
        for i, meta in enumerate(saved_meta):
            print("\n--- TF-IDF top terms for", meta[0], meta[1], "---")
            for term, score in tfidf_res.get(i, []):
                print(f"{term} ({score:.3f})")

    conn.close()
    print(f"SQLite DB saved at {DB_PATH}")

if __name__ == "__main__":
    main()
