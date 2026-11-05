"""
TP NÂ°3 - N-Gram Language Model (Part 1)

Streamlit application:
- Generate Unigrams, Bigrams, Trigrams and 4-grams.
- Work with one article, one category or the entire dataset.
- Apply normalization using Porter, Lancaster or Snowball stemming.

Run:
    python -m pip install streamlit pandas nltk
    python -m streamlit run app.py

Expected dataset structure:
    tp-tel/
        app.py
        bbc/
            business/
            entertainment/
            politics/
            sport/
            tech/
"""

import os
import re
import glob
from collections import Counter

import pandas as pd
import streamlit as st

from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="BBC News - N-gram LM",
    layout="wide"
)


# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_dataset(root: str) -> dict:
    """
    Load BBC News dataset.

    Returns:
        {
            category: [
                (filename, article_text),
                ...
            ]
        }
    """

    data = {}

    if not os.path.isdir(root):
        return data

    for cat in sorted(os.listdir(root)):

        cat_path = os.path.join(root, cat)

        if not os.path.isdir(cat_path):
            continue

        articles = []

        files = sorted(
            glob.glob(os.path.join(cat_path, "*.txt"))
        )

        for file_path in files:

            with open(
                file_path,
                "r",
                encoding="latin-1",
                errors="ignore"
            ) as file:

                content = file.read()

                if content.strip():
                    articles.append(
                        (os.path.basename(file_path), content)
                    )

        if articles:
            data[cat] = articles

    return data


# ---------------------------------------------------------
# TEXT PREPROCESSING
# ---------------------------------------------------------

def tokenize(text: str) -> list:
    """
    Convert text to lowercase and extract alphabetic tokens.
    """

    text = text.lower()

    return re.findall(r"[a-z]+", text)


def get_stemmer(name: str):
    """
    Return the selected stemming algorithm.
    """

    if name == "Porter":
        return PorterStemmer()

    elif name == "Lancaster":
        return LancasterStemmer()

    else:
        return SnowballStemmer("english")


def normalize(tokens: list, algo: str) -> list:
    """
    Apply stemming to tokens.
    """

    stemmer = get_stemmer(algo)

    return [
        stemmer.stem(token)
        for token in tokens
    ]


@st.cache_data(show_spinner=False)
def build_tokens(
    texts: tuple,
    use_norm: bool,
    algo: str
) -> list:
    """
    Concatenate tokens from all selected articles.
    Apply stemming if normalization is enabled.
    """

    tokens = []

    for text in texts:

        current_tokens = tokenize(text)

        if use_norm:
            current_tokens = normalize(
                current_tokens,
                algo
            )

        tokens.extend(current_tokens)

    return tokens


# ---------------------------------------------------------
# N-GRAM GENERATION
# ---------------------------------------------------------

def generate_ngrams(tokens: list, n: int) -> list:
    """
    Generate n-grams using consecutive tokens.
    """

    if len(tokens) < n:
        return []

    return list(
        zip(
            *[
                tokens[i:]
                for i in range(n)
            ]
        )
    )


# ---------------------------------------------------------
# N-GRAM LANGUAGE MODEL
# ---------------------------------------------------------

def ngram_model(tokens: list, n: int):
    """
    Generate the N-gram language model.

    Returns:
        DataFrame, total_ngrams, unique_ngrams

    Unigram:
        P(w) = C(w) / N

    Bigram, Trigram and 4-gram:
        P(wn | w1...wn-1)
        = C(w1...wn) / C(w1...wn-1)
    """

    grams = generate_ngrams(tokens, n)

    total = len(grams)

    if total == 0:
        return pd.DataFrame(), 0, 0

    # Count occurrences of each n-gram
    counts = Counter(grams)

    rows = []

    if n == 1:

        for gram, frequency in counts.items():

            probability = frequency / total

            rows.append(
                (gram[0], frequency, probability)
            )

    else:

        # Count occurrences of each context (prefix)
        context_counts = Counter()

        for gram, frequency in counts.items():

            context = gram[:-1]

            context_counts[context] += frequency

        # Calculate conditional probabilities
        for gram, frequency in counts.items():

            context = gram[:-1]

            probability = (
                frequency / context_counts[context]
            )

            rows.append(
                (
                    " ".join(gram),
                    frequency,
                    probability
                )
            )

    # Model name
    name = {
        1: "Unigram",
        2: "Bigram",
        3: "Trigram",
        4: "4-gram"
    }[n]

    # Create DataFrame
    df = pd.DataFrame(
        rows,
        columns=[
            name,
            "Frequency",
            "Probability P(Wn|W...)"
        ]
    )

    # Sort by frequency
    df = df.sort_values(
        ["Frequency", name],
        ascending=[False, True]
    ).reset_index(drop=True)

    # Round probabilities
    df["Probability P(Wn|W...)"] = (
        df["Probability P(Wn|W...)"].round(4)
    )

    return df, total, len(counts)


# ---------------------------------------------------------
# SIDEBAR - DATASET SETTINGS
# ---------------------------------------------------------

st.sidebar.header("Settings")

# Get the folder containing app.py
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# Default BBC dataset path
default_root = os.path.join(
    BASE_DIR,
    "bbc"
)

root = st.sidebar.text_input(
    "Dataset folder",
    value=default_root
)

# Load dataset
dataset = load_dataset(root)


# ---------------------------------------------------------
# CHECK DATASET
# ---------------------------------------------------------

if not dataset:

    st.title("BBC News")

    st.error(
        f"Dataset folder '{root}' not found or empty. "
        "Make sure the bbc folder contains category folders "
        "with .txt files."
    )

    st.stop()


# Get categories
categories = list(dataset.keys())

# Count all articles
n_articles_total = sum(
    len(articles)
    for articles in dataset.values()
)


# ---------------------------------------------------------
# SIDEBAR - DATA SELECTION
# ---------------------------------------------------------

data_mode = st.sidebar.radio(
    "Data",
    [
        "One article",
        "One category",
        "Entire dataset"
    ],
    index=1
)

# Initialize selected category safely
selected_category = categories[0]
article_idx = 1

if data_mode in ("One article", "One category"):

    selected_category = st.sidebar.selectbox(
        "Category",
        categories,
        index=0
    )


# Article selection
if data_mode == "One article":

    max_idx = len(dataset[selected_category])

    article_idx = st.sidebar.number_input(
        "Article number",
        min_value=1,
        max_value=max_idx,
        value=1,
        step=1
    )


# ---------------------------------------------------------
# SIDEBAR - NORMALIZATION SETTINGS
# ---------------------------------------------------------

st.sidebar.divider()

stem_algo = st.sidebar.selectbox(
    "Stemming algorithm",
    [
        "Porter",
        "Lancaster",
        "Snowball"
    ]
)

use_norm = st.sidebar.checkbox(
    "Use Normalized Text"
)


# ---------------------------------------------------------
# SELECT TEXTS
# ---------------------------------------------------------

if data_mode == "One article":

    fname, article_text = dataset[selected_category][
        int(article_idx) - 1
    ]

    texts = (article_text,)

    volume_label = (
        f"{selected_category}/{fname}"
    )


elif data_mode == "One category":

    texts = tuple(
        text
        for _, text in dataset[selected_category]
    )

    volume_label = (
        f"category '{selected_category}' "
        f"({len(texts)} articles)"
    )


else:

    texts = tuple(
        text
        for articles in dataset.values()
        for _, text in articles
    )

    volume_label = (
        f"entire dataset ({len(texts)} articles)"
    )


# ---------------------------------------------------------
# MAIN PAGE
# ---------------------------------------------------------

st.title("BBC News")

st.markdown(
    f"Contains **{n_articles_total} articles**, "
    f"divided into **{len(categories)} categories**: "
    + ", ".join(
        f"**{category.capitalize()}**"
        for category in categories
    )
)

st.caption(
    f"Current volume: {volume_label}"
)


# ---------------------------------------------------------
# PART 1 - TEXT DISPLAY
# ---------------------------------------------------------

with st.expander("Part 1 - Text Display"):

    if data_mode == "One article":

        st.text(texts[0])

    else:

        preview_n = st.slider(
            "Number of articles to preview",
            min_value=1,
            max_value=min(10, len(texts)),
            value=1
        )

        for i, text in enumerate(
            texts[:preview_n],
            start=1
        ):

            st.markdown(
                f"**Article {i}**"
            )

            st.text(
                text[:1500]
                + ("..." if len(text) > 1500 else "")
            )


# ---------------------------------------------------------
# PART 2 - TEXT PREPROCESSING
# ---------------------------------------------------------

with st.expander("Part 2 - Text Preprocessing"):

    sample = texts[0]

    # Tokenization
    raw_tokens = tokenize(sample)

    st.write(
        f"**Tokens (lowercase, alphabetic only)** "
        f"- {len(raw_tokens)} tokens in first text"
    )

    st.write(raw_tokens[:100])

    # Stemming
    normalized_tokens = normalize(
        raw_tokens,
        stem_algo
    )

    st.write(
        f"**After {stem_algo} stemming**"
    )

    st.write(normalized_tokens[:100])

    # Token counts
    col1, col2 = st.columns(2)

    col1.metric(
        "Original tokens",
        len(raw_tokens)
    )

    col2.metric(
        "Normalized tokens",
        len(normalized_tokens)
    )


# ---------------------------------------------------------
# PART 3 - N-GRAM LANGUAGE MODEL
# ---------------------------------------------------------

with st.expander(
    "Part 3 - N-gram Language Model",
    expanded=True
):

    c1, c2, _ = st.columns([1, 1, 4])

    model_name = c1.selectbox(
        "Select N-gram Model",
        [
            "Unigram",
            "Bigram",
            "Trigram",
            "4-gram"
        ]
    )

    generate = c2.button(
        "Generate ...",
        type="primary"
    )

    n = {
        "Unigram": 1,
        "Bigram": 2,
        "Trigram": 3,
        "4-gram": 4
    }[model_name]

    if generate:

        with st.spinner("Computing n-grams..."):

            # Build tokens
            tokens = build_tokens(
                texts,
                use_norm,
                stem_algo
            )

            # Generate model
            df, total, unique = ngram_model(
                tokens,
                n
            )

        if df.empty:

            st.warning(
                "Not enough tokens to build this model."
            )

        else:

            # Display results
            st.dataframe(
                df,
                use_container_width=True,
                height=300
            )

            label = model_name

            st.info(
                f"**{total}** {label}s with "
                f"**{unique}** unique {label}s"
            )

            # Download CSV
            normalization_label = (
                f"norm_{stem_algo.lower()}"
                if use_norm
                else "raw"
            )

            filename = (
                f"{model_name.lower()}_"
                f"{normalization_label}.csv"
            )

            st.download_button(
                "Download CSV",
                df.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name=filename,
                mime="text/csv"
            )