import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')

def eos_and_noun_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    words = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    doc = nlp(sentence)

    # Map token indices from tokenizer to spaCy
    spacy_to_token_indices = {}
    for token in doc:
        word_pieces = tokenizer.tokenize(token.text)
        for idx in range(len(words) - len(word_pieces) + 1):
            if words[idx:idx+len(word_pieces)] == word_pieces:
                for wp, token_idx in zip(word_pieces, range(idx, idx+len(word_pieces))):
                    spacy_to_token_indices[(token.idx, token.idx + len(token.text))] = token_idx

    for token in doc:
        if token.pos_ in ("NOUN", "PROPN"):
            if (token.idx, token.idx + len(token.text)) in spacy_to_token_indices:
                token_idx = spacy_to_token_indices[(token.idx, token.idx + len(token.text))]
                out[token_idx, token_idx] = 1

    # End of sentence focus
    out[-2, -2] = 1 # EOS token in Roberta

    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize return matrix

    return "End-of-Sentence and Significant Noun Focus Pattern", out