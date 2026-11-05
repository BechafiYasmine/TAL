"""
TP NÂ°2 - Text Preprocessing (Part 2)
Part I  : Text visualization with WordCloud (one article / category / dataset)
Part II : Sentence segmentation (?, ., !, \\n) + sentence count
"""
from wordcloud import WordCloud
from tp1 import split_sentences, tokenize, BOS, EOS


# ----------------------------------------------------------------------------
# I. TEXT VISUALIZATION
# ----------------------------------------------------------------------------
def generate_wordcloud(text, width=1000, height=500, background="white"):
    """Return a numpy image array of the word cloud generated from `text`."""
    # Clean text: remove sentence markers so they do not appear in the cloud
    text = text.replace(BOS, " ").replace(EOS, " ")
    wc = WordCloud(width=width, height=height, background_color=background,
                   collocations=False).generate(text)
    return wc.to_array()


# ----------------------------------------------------------------------------
# II. SENTENCE SEGMENTATION
# ----------------------------------------------------------------------------
def segment_sentences(text):
    """Return the list of sentences, formatted as: <s> ... </s> (lowercased, no final punctuation)."""
    sentences = []
    for sent in split_sentences(text):
        toks = [t for t in tokenize(sent) if t not in (".", "?", "!")]
        if toks:
            sentences.append(f"{BOS} {' '.join(toks)} {EOS}")
    return sentences


def count_sentences(sentences):
    return len(sentences)