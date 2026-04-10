import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

# Load spacy model
nlp = spacy.load('en_core_web_sm')

def noun_phrase_base_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> str:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    doc = nlp(sentence)

    # Dictionary to align tokenizer and spacy token indices
    align_dictionary = {}
    tokenize_str = toks.tokens()[:]

    i, j = 0, 0
    while i < len(doc) and j < len(tokenize_str):
        if doc[i].text == tokenize_str[j] or tokenize_str[j].startswith(doc[i].text):
            align_dictionary[j] = i
            j += 1
        elif tokenize_str[j].startswith("\u"):  # Handling possible subword tokens
            j += 1
            continue
        else:
            align_dictionary[j] = i
            i += 1
            j += 1

    for np in doc.noun_chunks:
        indices = [i for i, tok in align_dictionary.items() if np.start <= tok < np.end]
        for idx in indices:
            out[idx, idx] = 1
            for other_idx in indices:
                if idx != other_idx:
                    out[idx, other_idx] = 1

    out[0, 0] = 1  # Start token
    out[-1, 0] = 1 # End token

    # Normalize attention matrix
    out += 1e-4
    out /= out.sum(axis=1, keepdims=True)

    return "Noun Phrase Base Attention", out