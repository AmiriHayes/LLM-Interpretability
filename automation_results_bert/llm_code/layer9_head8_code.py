import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')

def object_centric_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    words = sentence.split()
    doc = nlp(sentence)

    # Construct a mapping to align spaCy tokens with tokenized input
    spacy_to_tokenizer_map = {}
    word_idx = 1  # Assuming [CLS] token is at index 0

    for token in doc:
        token_text = tokenizer.decode(toks.input_ids[0][word_idx])
        spacy_to_tokenizer_map[token.i] = word_idx
        # Adjust for subtokens, not just first token
        word_idx += len(tokenizer(token.text)['input_ids']) - 2

    # Identify noun tokens in spaCy doc
    object_indices = [spacy_to_tokenizer_map[token.i] for token in doc if token.pos_ == "NOUN"]

    # Create attention pattern centered around noun tokens
    for idx in object_indices:
        out[idx, idx] = 1
        if idx-1 > 0:  # left context attention
            out[idx, idx-1] = 0.5
        if idx+1 < len_seq:  # right context attention
            out[idx, idx+1] = 0.5

    # Ensure no row is all zeros by assigning attention to SEP
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Avoid division by zero and normalize
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Object-Centric Attention", out