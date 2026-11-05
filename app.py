"""
Streamlit interface - TP1 + TP2 + TP3

Run:
    streamlit run app.py

Individual TPs:
    streamlit run tp1.py
    streamlit run tp2.py
    streamlit run tp3.py
"""

import streamlit as st
import tp1
import tp2
import tp3_model

st.set_page_config(page_title="TP - NLP", layout="wide")

tp1.download_nltk()
data = tp1.ensure_data()


# ------------------------------ SIDEBAR -------------------------------------

with st.sidebar:

    st.title("NLP Projects")

    category = st.selectbox(
        "Select Category",
        [c.capitalize() for c in data]
    ).lower()

    number = st.number_input(
        "Number of article",
        min_value=1,
        max_value=len(data[category]),
        value=1,
        step=1,
        key=f"num_{category}"
    )

    level = st.radio(
        "Data",
        ["One article", "One category", "Entire dataset"],
        index=0
    )

    st.divider()

    algo = st.selectbox(
        "Stemming algorithm",
        list(tp1.STEMMERS)
    )

    use_norm = st.checkbox("Use Normalized Text")

# ------------------------------ HEADER --------------------------------------

st.title("TP - Natural Language Processing (NLP)")
st.header("BBC News")

st.markdown(
    f"Contains **{tp1.count_articles(data)} articles**, divided into 5 categories: "
    "**Business, Entertainment, Politics, Sport and Tech** "
    "- published during **2004-2005**"
)


# ------------------------------ TEXT ----------------------------------------

raw_text = tp1.get_text(data, level, category, number)

text = (
    tp1.normalize_text(raw_text, algo)
    if use_norm
    else raw_text
)


# ------------------------------ PART 1 (TP1) --------------------------------

with st.expander("Part 1 — Text Display", expanded=False):

    if st.button("Display Text"):
        st.text_area("Content", raw_text, height=350)


# ------------------------------ PART 2 (TP1 + TP2) --------------------------

with st.expander("Part 2 — Text Preprocessing", expanded=True):

    c1, c2, c3, c4, c5 = st.columns(5)

    if c1.button("Apply Tokenization"):
        st.session_state.action = "tokenize"

    example = c2.checkbox("Tokenization example")

    if c3.button("Apply Normalization"):
        st.session_state.action = "normalize"

    if c4.button("Word Cloud"):
        st.session_state.action = "wordcloud"

    if c5.button("Sentence Segmentation"):
        st.session_state.action = "sentences"

    action = st.session_state.get("action")

    source = tp1.TOKENIZATION_EXAMPLE if example else text

    if action == "tokenize":
        tp1.render_tokenization(source, example)

    elif action == "normalize":
        tp1.render_normalization(source, algo)

    elif action == "wordcloud":
        st.image(
            tp2.generate_wordcloud(source),
            use_container_width=True
        )

    elif action == "sentences":

        sentences = tp2.segment_sentences(source)

        a, b = st.columns(2)

        a.text_area("Text", source, height=300)

        b.text_area(
            "Sentences",
            "\n".join(sentences),
            height=300
        )

        b.markdown(
            f"**{tp2.count_sentences(sentences)} Sentences**"
        )

# ------------------------------ PART 3 (TP3) -------------------------------
tp3_model.render_ngram_model(
    raw_text,
    use_norm,
    algo,
    (level, category, int(number)),
)