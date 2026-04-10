import numpy as np
from transformers import PreTrainedTokenizerBase

def cross_reference_linked_concepts(sentence: str, tokenizer: PreTrainedTokenizerBase) -> str:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Use the attention patterns observed to encode relations between concepts
    token_pairs = [
        ("fuel", "healthy"), ("play", "fast"), ("sharing", "helping"), ("shirt", "button"),
        ("tree", "leaves"), ("happy", "health"), ("fuel", "needed"), ("share", "shirt"),
        ("girl", "lily"), ("named", "")]

    # First, create a token to index map
    token_to_index = {token.text: index for index, token in enumerate(toks[0].ids)}

    # Capture relationships based on the observed attention patterns
    for word1, word2 in token_pairs:  
        if word1 in token_to_index and word2 in token_to_index:
            idx1 = token_to_index[word1]
            idx2 = token_to_index[word2]
            out[idx1, idx2] = 1
            out[idx2, idx1] = 1

    # Handle special tokens CLS and SEP
    out[0, 0] = 1  # CLS token self-attention
    out[-1, -1] = 1  # SEP token self-attention

    # Ensure no row is entirely zero by giving minimum attention (to SEP)
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the matrix rows
    out = out / out.sum(axis=1, keepdims=True)

    return "Cross-Reference of Linked Concepts Pattern", out