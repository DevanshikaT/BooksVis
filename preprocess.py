import pandas as pd
import numpy as np
from textblob import TextBlob
import json
from pathlib import Path

# Input CSV (xz compressed) path
CSV_PATH = 'GoodReads_100k_books.csv.xz'
DATA_DIR = Path('data')

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True)

# Load dataset (pandas can read xz directly)
df = pd.read_csv(CSV_PATH, compression='xz')

# Derive helper columns for author aggregates
author_stats = df.groupby('author').agg(
    titles_per_author=('title', 'count'),
    author_median_rating=('rating', 'median')
)
df = df.merge(author_stats, left_on='author', right_index=True)

# Derive polarity500 using TextBlob on first 500 characters of description
def sentiment(text):
    if isinstance(text, str) and text:
        return TextBlob(text[:500]).sentiment.polarity
    return np.nan

print('Computing polarity500...')
df['polarity500'] = df['desc'].map(sentiment)

# log10 of totalratings
df['log10_totalratings'] = np.log10(df['totalratings'].replace(0, np.nan))

# Stories configuration
stories = {
    'pages_vs_rating': ('pages', 'rating'),
    'author_output_vs_rating': ('titles_per_author', 'author_median_rating'),
    'desc_sentiment_vs_rating': ('polarity500', 'rating'),
    'popularity_vs_rating': ('log10_totalratings', 'rating'),
}

np.random.seed(42)

for name, (xcol, ycol) in stories.items():
    print(f'Processing {name}...')
    sub = df[[xcol, ycol]].dropna()
    # Outlier trimming
    q1 = sub.quantile(0.25)
    q3 = sub.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    trimmed = sub[(sub[xcol] >= lower[xcol]) & (sub[xcol] <= upper[xcol]) &
                  (sub[ycol] >= lower[ycol]) & (sub[ycol] <= upper[ycol])]
    # Down-sample if necessary
    if len(trimmed) > 10000:
        trimmed = trimmed.sample(10000, random_state=42)
    trimmed = trimmed.round(3)

    records = trimmed.rename(columns={xcol: 'x', ycol: 'y'}).to_dict('records')

    # Increase rounding if file too large
    for decimals in [3, 2, 1]:
        data_json = json.dumps(records, separators=(',', ':'))
        if len(data_json.encode('utf-8')) <= 300_000:
            break
        # re-round with fewer decimals
        trimmed = trimmed.round(decimals)
        records = trimmed.rename(columns={xcol: 'x', ycol: 'y'}).to_dict('records')

    # Write JSON
    out_path = DATA_DIR / f'{name}.json'
    out_path.write_text(data_json)
    size_kb = out_path.stat().st_size / 1024
    print(f'Wrote {out_path} ({size_kb:.1f} KB)')
