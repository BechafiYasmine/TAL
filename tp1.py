"""
TP N°1 - Text Preprocessing
Part I  : Text display (one article / one category / entire dataset)
Part II : Tokenization (RegexpTokenizer) + Stemming (Porter, Lancaster, Snowball)
"""
import os
import re
import nltk
import streamlit as st
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer

DATA_DIR = "bbc"          # folder containing business/ entertainment/ politics/ sport/ tech/
BOS, EOS = "<s>", "</s>"

# ----------------------------------------------------------------------------
# I. TEXT DISPLAY
# ----------------------------------------------------------------------------
CATEGORIES = {"business", "entertainment", "politics", "sport", "tech"}


def find_dataset_root(start=None, max_depth=4):
    """Search for the folder that directly contains the category folders
    (business, entertainment, ...), starting from `start`, the script folder, and the cwd."""
    roots = [start] if start else []
    roots += [os.path.dirname(os.path.abspath(__file__)), os.getcwd()]
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        base_depth = root.rstrip(os.sep).count(os.sep)
        for dirpath, dirnames, _ in os.walk(root):
            if dirpath.count(os.sep) - base_depth >= max_depth:
                dirnames[:] = []
                continue
            if CATEGORIES.issubset({d.lower() for d in dirnames}):
                return dirpath
    return None


def load_dataset(data_dir=DATA_DIR):
    """Return {category: [article_text, ...]} read from <root>/<category>/*.txt"""
    root = find_dataset_root(data_dir if os.path.isdir(data_dir) else None)
    data = {}
    if root is None:
        return data
    for cat in sorted(os.listdir(root)):
        cat_path = os.path.join(root, cat)
        if cat.lower() not in CATEGORIES or not os.path.isdir(cat_path):
            continue
        articles = []
        for fname in sorted(os.listdir(cat_path)):
            if fname.endswith(".txt"):
                with open(os.path.join(cat_path, fname), "r", encoding="latin-1") as f:
                    articles.append(f.read().strip())
        if articles:
            data[cat.lower()] = articles
    return data


def get_one_article(data, category, number):
    """number is 1-based (as in the interface)."""
    arts = data[category]
    number = max(1, min(number, len(arts)))
    return arts[number - 1]


def get_one_category(data, category):
    return "\n\n".join(data[category])


def get_entire_dataset(data):
    return "\n\n".join(get_one_category(data, c) for c in data)


def get_text(data, level, category=None, number=1):
    """level in {'One article', 'One category', 'Entire dataset'}"""
    if level == "One article":
        return get_one_article(data, category, number)
    if level == "One category":
        return get_one_category(data, category)
    return get_entire_dataset(data)


def count_articles(data):
    return sum(len(v) for v in data.values())


# ----------------------------------------------------------------------------
# II-A. TOKENIZATION
# ----------------------------------------------------------------------------
TOKEN_REGEX = r"""
    (?:[A-Za-z]\.){2,}                              # abbreviations: U.S.T.H.B.  u.S.T.h.b.
  | [£$€]?\d+(?:[.,\-]\d+)*(?:[%€]|[A-Za-z]+)?      # numbers: 2020 2020.45 2020-45 2020,45 £347 600€ 2020A $1.13bn 76%
  | [A-Za-z]\w*(?:[@\-]\w+(?:[.,]\d+)*)*            # words: A2020 Precision@10 Precision-453.45 State-of-the-art
  | [.?!]                                           # sentence punctuation kept as tokens
"""
_tokenizer = RegexpTokenizer(TOKEN_REGEX, flags=re.VERBOSE)


def tokenize(text):
    """Lowercase + regexp tokenization (punctuation . ? ! kept as tokens)."""
    return _tokenizer.tokenize(text.lower())


def split_sentences(text):
    """Sentence boundaries: ?  .  !  and newline (\\n)."""
    parts = re.split(r"(?<=[.?!])\s+|\n+", text)
    return [p.strip() for p in parts if p and p.strip()]


def tokenize_with_markers(text):
    """Tokenize each sentence, drop the sentence-final . ? ! and add <s> ... </s>."""
    tokens = []
    for sent in split_sentences(text):
        toks = [t for t in tokenize(sent) if t not in (".", "?", "!")]
        if toks:
            tokens += [BOS] + toks + [EOS]
    return tokens


def vocabulary(tokens):
    return sorted(set(tokens))


TOKENIZATION_EXAMPLE = (
    "This example provides a few guidelines to help you prepare your regular expressions. "
    "Abbreviations such as U.S.T.H.B., u.s.t.h.b., or u.S.T.h.b. should be properly handled. "
    "Certain metric names should also be recognized, for example: Precision@10, precision@140.45, "
    "Precision-453, or Precision-453.45. In addition, combinations of letters and digits like "
    "A2020, 2020A, £347 or 600€ should be taken into account. Parentheses are not considered in "
    "this context — for example: (DZ) or (AI). You should also handle numeric combinations such as "
    "2020, 2020.45, 2020-45, or 2020,45, as well as mixed character sequences like State-of-the-art. "
    "Finally, remember that combinations of characters are perfectly fine — after all, most of this "
    "text is made up of them ;) and don’t forget to check the list of tokens extracted from this text."
)

# ----------------------------------------------------------------------------
# II-B. STEMMING (NORMALIZATION)
# ----------------------------------------------------------------------------
STEMMERS = {
    "Porter": PorterStemmer,
    "Lancaster": LancasterStemmer,
    "Snowball": lambda: SnowballStemmer("english"),
}


def stem_tokens(tokens, algorithm="Porter"):
    stemmer = STEMMERS[algorithm]()
    return [t if t in (BOS, EOS) else stemmer.stem(t) for t in tokens]


def normalize_text(text, algorithm="Porter"):
    """Return the text with every sentence stemmed (used by 'Use Normalized Text')."""
    out = []
    for sent in split_sentences(text):
        toks = [t for t in tokenize(sent) if t not in (".", "?", "!")]
        out.append(" ".join(stem_tokens(toks, algorithm)))
    return "\n".join(s for s in out if s)


def download_nltk():
    for pkg in ("punkt", "punkt_tab", "stopwords"):
        nltk.download(pkg, quiet=True)


# ----------------------------------------------------------------------------
# INTERFACE (TP1)  ->  streamlit run tp1.py
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _cached_load(path):
    return load_dataset(path)


def ensure_data():
    """Load the dataset or ask the user for its folder."""
    data = _cached_load(DATA_DIR)
    if not data:
        st.error("Dataset not found. Enter the path of the folder that contains "
                 "business / entertainment / politics / sport / tech.")
        custom = st.text_input("Dataset folder path", placeholder="D:\\path\\to\\bbc")
        if custom:
            data = _cached_load(custom)
        if not data:
            st.stop()
    return data


def render_tokenization(source, example=False):
    """'Apply Tokenization' output. Example -> Text | Tokens (python list);
    otherwise -> Tokens | Vocabulary | Text with counts."""
    if example:
        tokens = tokenize(source)
        a, b = st.columns(2)
        a.text_area("Text", source, height=300)
        b.text_area("Tokens", str(tokens), height=300)
        return
    tokens = tokenize_with_markers(source)
    vocab = vocabulary(tokens)
    a, b, c = st.columns([1, 1, 2])
    a.text_area("Tokens", "\n".join(tokens), height=300)
    b.text_area("Vocabulary", "\n".join(vocab), height=300)
    c.text_area("Text", source, height=300)
    a.markdown(f"**{len(tokens)}** Tokens")
    b.markdown(f"**{len(vocab)}** Unique words")


def render_normalization(source, algo):
    """'Apply Normalization' output: stemmed tokens | original tokens | vocabulary | text."""
    tokens = tokenize_with_markers(source)
    stemmed = stem_tokens(tokens, algo)
    vocab = vocabulary(stemmed)
    a, b, c, d = st.columns(4)
    a.text_area("Tokens", "\n".join(stemmed), height=300)
    b.text_area("Tokens before stemming", "\n".join(tokens), height=300)
    c.text_area("Vocabulary", "\n".join(vocab), height=300)
    d.text_area("Text", source, height=300)
    a.markdown(f"**{len(stemmed)}** Tokens normalized")
    b.markdown(f"**{len(tokens)}** Tokens")
    c.markdown(f"**{len(vocab)}** Unique words")


def run_ui():
    st.set_page_config(page_title="TP1 - NLP", layout="wide")
    download_nltk()
    data = ensure_data()

    with st.sidebar:
        category = st.selectbox("Select Category", [c.capitalize() for c in data]).lower()
        number = st.number_input("Number of article", min_value=1, max_value=len(data[category]),
                                 value=1, step=1, key=f"num_{category}")
        level = st.radio("Data", ["One article", "One category", "Entire dataset"], index=0)
        st.divider()

    st.title("TP - Natural Language Processing (NLP)")
    st.header("BBC News")
    st.markdown(f"Contains **{count_articles(data)} articles**, divided into 5 categories: "
                "**Business, Entertainment, Politics, Sport** & **Tech** - published during **2004-2005**")

    raw = get_text(data, level, category, number)

    # ---- Part 1 : text display ----
    if st.button("Display Text"):
        st.session_state.show_text = True
    if st.session_state.get("show_text"):
        with st.expander("Part 1 — Text Display", expanded=True):
            st.text_area("Content", raw, height=350)

    # ---- Part 2 : preprocessing ----
    left, right = st.columns(2)
    with left:
        if st.button("Apply Tokenization"):
            st.session_state.action = "tokenize"
        example = st.checkbox("Tokenization example")
    with right:
        if st.button("Apply Normalization"):
            st.session_state.action = "normalize"
        algo = st.selectbox("Stemming algorithm", list(STEMMERS))

    source = TOKENIZATION_EXAMPLE if example else raw
    action = st.session_state.get("action")
    if action:
        with st.expander("Part 2 — Text Preprocessing", expanded=True):
            if action == "tokenize":
                render_tokenization(source, example)
            else:
                render_normalization(source, algo)


if __name__ == "__main__":
    run_ui()