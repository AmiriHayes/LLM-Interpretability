from transformers import PreTrainedTokenizerBase
import numpy as np
from typing import Tuple

# The function predicts the attention pattern for Layer 3, Head 3, which emphasizes the main topic noun in a sentence.
def topic_centralization(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize sentence using spaCy to map main noun detection
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)

    # Identify the main subject or topic in the sentence based on the following criteria: 
    # Choose token representations of nouns higher up in the syntactic tree (heuristic here is just the first noun).
    topic_token_index = None
    for token in doc:
        if token.pos_ == 'NOUN' or token.dep_ == 'nsubj':
            topic_token_index = token.i + 1
            break

    # Assign high attention weights to the detected main topic
    if topic_token_index is not None:
        for i in range(1, len_seq - 1):
            out[i, topic_token_index] = 1

    # Normalize the attention output to prevent division by zero
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Ensure no row is entirely zeros and allow stable division
    out = out / out.sum(axis=1, keepdims=True)  # Normalize

    return "Topic Centralization Pattern", out