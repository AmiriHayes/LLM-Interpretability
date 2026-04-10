import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')

def temporal_marker_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> (str, np.ndarray):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize sentence using spaCy
    doc = nlp(sentence)

    # Identify temporal expressions in the sentence
    temporal_markers = {'before', 'after', 'until', 'since', 'whenever', 'when', 'while', 'during', 'entire', 'quickly', 'suddenly', 'gradually', 'immediately', 'simultaneously'}

    # Align token indices between tokenizer and spaCy
    word_indices_mapping = {}
    bpe_cursor = 0
    for tok_spacy in doc:
        while (bpe_cursor < len_seq-1 and toks.word_ids(batch_index=0)[bpe_cursor] is None):
            bpe_cursor += 1
        if bpe_cursor < len_seq-1:
            word_indices_mapping[tok_spacy.i] = bpe_cursor
            bpe_cursor += 1

    for token in doc:
        if token.text.lower() in temporal_markers:
            token_idx = word_indices_mapping.get(token.i)
            if token_idx:
                for j in range(len_seq):
                    out[token_idx, j] = 1

    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the output matrix by rows
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Temporal Marker Attention", out