import numpy as np
from transformers import PreTrainedTokenizerBase, RobertaTokenizer
from typing import Tuple
import spacy

# Load the tokenizer and spacy model needed
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
nlp = spacy.load('en_core_web_sm')


def transition_and_conjunctions_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize using spacy for specific linguistic features
    doc = nlp(sentence)

    # Match spaCy tokens to tokenizer tokens
    spacy_to_tokenizer_map = {}
    j = 0
    for i, tok in enumerate(toks.input_ids[0]):
        while j < len(doc) and doc[j].text != tokenizer.decode(tok.item()).strip():
            j += 1
        if j < len(doc):
            spacy_to_tokenizer_map[i] = j

    # List conjunction-like and transition words of interest
    conjunctions = {"and", "or", "but", "so", "because", "although", "if", "when"}
    doc_tokens = [tok.text.lower() for tok in doc]

    # Apply attention
    for i, token in enumerate(doc_tokens):
        if token in conjunctions or token == ',':
            if i in spacy_to_tokenizer_map.values():
                out[spacy_to_tokenizer_map[i], :] = 1.0
                out[:, spacy_to_tokenizer_map[i]] = 1.0

    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize to ensure valid attention distribution
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Sentence Transition Points and Conjunctions", out