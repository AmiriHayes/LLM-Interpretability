import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')

# Initialize SpaCy for linguistic processing.
def sentiment_intensity_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize with spacy for alignment with the sentiment analysis
doc = nlp(sentence)

    # Find intensity and sentiment words using spaCy's weights
    intensity_words = []
    for tok in doc:
        if tok.pos_ in ['ADV', 'ADJ']:  # Adverbs and adjectives can imply intensity
            intensity_words.append(tok.i)

    # Assign predicted patterns based on intensity. The matrix signifies heavier attention between tokens
    for idx in intensity_words:
        for word in doc:
            if (word.i != idx):
                out[idx + 1, word.i + 1] = 1  # Intensity word attends to others

    # Assign self-attention for CLS and SEP tokens
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize the matrix rows so they sum to 1
    out = out / out.sum(axis=1, keepdims=True, where=out.sum(axis=1)!=0)
    return "Intensity and Sentiment Connection Pattern", out