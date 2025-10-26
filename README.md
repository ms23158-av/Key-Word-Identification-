

# Keyword Identification Project

Automated keyword extraction and visualization from institutional websites using Python NLP, web scraping, and data analysis.  
Analyze, compare, and visualize the main topics and keyword structures of sites like IIM Ahmedabad and IISER Mohali.

***

## Features

- Web scraping, text cleaning, and lemmatization
- Keyword frequency analysis per site
- TF-IDF feature extraction highlighting distinctive terms
- Visualizations: bar charts, word clouds, heatmaps, and co-occurrence networks
- SQLite storage for efficient text and metadata management

***

## Example Outputs

_See the outputs below for the kind of results this project generates:_  
```md
Top Words Bar Charts
[IIM Ahmedabad Bar Chart](IIM_Ahmedabad_bar_chart.jpeg)
[IISER Mohali Bar Chart](IISER_Mohali_bar_chart.jpeg)

Word Clouds
[IIM Ahmedabad Word Cloud](IIM_Ahmedabad_wordcloud.jpeg)
[IISER Mohali Word Cloud](IISER_Mohali_wordcloud.jpeg)

TF-IDF Heatmap
[TF-IDF Heatmap](tfidf_heatmap.jpeg)

Word Co-occurrence Networks
[IIM Ahmedabad Network](IIM_Ahmedabad_network.jpeg)
[IISER Mohali Network](IISER_Mohali_network.jpeg)
```

***

## Getting Started

### Prerequisites

- Python 3.x

Install all dependencies using:

```bash
pip install -r requirements.txt
```

#### requirements.txt

```txt
newspaper3k
requests
beautifulsoup4
tqdm
scikit-learn
lxml
seaborn
nltk
pandas
matplotlib
networkx
wordcloud
```

***

### Usage

1. Clone this repository.
2. Run the main script to scrape sites, analyze keywords, and generate results:

    ```bash
    python main.py
    ```

This will:
- Crawl and analyze a set number of pages (`MAX_PAGES` variable).
- Generate all result images in the working directory.

Optional: For automatic git commit and push (Windows), use:
```bash
submit.bat
```

***

## Code Overview

- **main.py**: Complete workflow, including scraping with BeautifulSoup and requests, text processing (NLTK), and all visualizations (matplotlib, seaborn, wordcloud, networkx).
- **requirements.txt**: Lists all Python library dependencies.
- **submit.bat**: (Optional) Batch file for auto-committing and pushing changes via git.

***

## How it Works

1. **Crawling**: Scrapes a limited number of internal pages for each institution.
2. **Text Processing**: Cleans, tokenizes, removes stopwords, and lemmatizes content.
3. **Database**: Stores page data in a local SQLite database for easy re-use.
4. **Analysis**:
   - Most common keywords (by frequency)
   - TF-IDF feature extraction for distinctive terms
5. **Visualization**:
   - Bar charts and word clouds for prominent keywords
   - TF-IDF heatmap for top discriminatory terms
   - Network graph of word co-occurrences

***

## Customization

- Change `MAX_PAGES` in `main.py` to scrape more pages (be considerate of server load).
- Add additional institutional URLs in the `SITES` dictionary in `main.py`.

***


## Acknowledgments

Built with open-source libraries including BeautifulSoup, requests, nltk, pandas, matplotlib, seaborn, wordcloud, and networkx.

