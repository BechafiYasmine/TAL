"""Import-safe n-gram language-model helpers and renderer for the combined app."""

import re
from collections import Counter

import pandas as pd
import streamlit as st
from nltk.stem import LancasterStemmer, PorterStemmer, SnowballStemmer


def tokenize(text: str) -> list:
    """Convert text to lowercase alphabetic tokens."""
    return re.findall(r"[a-z]+", text.lower())


def normalize(tokens: list, algo: str) -> list:
    """Apply the selected stemming algorithm."""
    if algo == "Porter":
        stemmer = PorterStemmer()
    elif algo == "Lancaster":
        stemmer = LancasterStemmer()
    else:
        stemmer = SnowballStemmer("english")
    return [stemmer.stem(token) for token in tokens]


@st.cache_data(show_spinner=False)
def build_tokens(texts: tuple, use_norm: bool, algo: str) -> list:
    """Concatenate selected text tokens and optionally stem them."""
    tokens = []
    for text in texts:
        current_tokens = tokenize(text)
        if use_norm:
            current_tokens = normalize(current_tokens, algo)
        tokens.extend(current_tokens)
    return tokens


def generate_ngrams(tokens: list, n: int) -> list:
    """Generate n-grams using consecutive tokens."""
    if len(tokens) < n:
        return []
    return list(zip(*(tokens[i:] for i in range(n))))


def ngram_model(tokens: list, n: int):
    """Return the n-gram probability table, total count, and unique count."""
    grams = generate_ngrams(tokens, n)
    total = len(grams)
    if total == 0:
        return pd.DataFrame(), 0, 0

    counts = Counter(grams)
    rows = []
    if n == 1:
        for gram, frequency in counts.items():
            rows.append((gram[0], frequency, frequency / total))
    else:
        context_counts = Counter()
        for gram, frequency in counts.items():
            context_counts[gram[:-1]] += frequency
        for gram, frequency in counts.items():
            rows.append((
                " ".join(gram),
                frequency,
                frequency / context_counts[gram[:-1]],
            ))

    name = {1: "Unigram", 2: "Bigram", 3: "Trigram", 4: "4-gram"}[n]
    df = pd.DataFrame(
        rows,
        columns=[name, "Frequency", "Probability P(Wn|W...)"],
    )
    df = df.sort_values(
        ["Frequency", name],
        ascending=[False, True],
    ).reset_index(drop=True)
    df["Probability P(Wn|W...)"] = df["Probability P(Wn|W...)"].round(4)
    return df, total, len(counts)


def render_ngram_model(text: str, use_norm: bool, algo: str, scope_key: tuple):
    """Render TP3 using the active TP1/TP2 text selection."""
    with st.expander("Part 3 — N-gram Language Model", expanded=False):
        controls, _ = st.columns([1, 4])
        model_name = controls.selectbox(
            "Select N-gram Model",
            ["Unigram", "Bigram", "Trigram", "4-gram"],
            key="tp3_model",
        )
        generate = controls.button(
            "Generate n-grams",
            type="primary",
            key="tp3_generate",
        )

        config = (scope_key, use_norm, algo, model_name)
        if generate:
            n = {
                "Unigram": 1,
                "Bigram": 2,
                "Trigram": 3,
                "4-gram": 4,
            }[model_name]
            with st.spinner("Computing n-grams..."):
                tokens = build_tokens((text,), use_norm, algo)
                result = ngram_model(tokens, n)
            st.session_state["tp3_result"] = result
            st.session_state["tp3_result_config"] = config

        if st.session_state.get("tp3_result_config") != config:
            return

        df, total, unique = st.session_state["tp3_result"]
        if df.empty:
            st.warning("Not enough tokens to build this model.")
            return

        st.dataframe(df, width="stretch", height=300)
        st.info(f"**{total}** {model_name.lower()}s with **{unique}** unique")
        normalization_label = f"norm_{algo.lower()}" if use_norm else "raw"
        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name=f"{model_name.lower()}_{normalization_label}.csv",
            mime="text/csv",
            key="tp3_download",
        )
