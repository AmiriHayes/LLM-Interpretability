import numpy as np
from transformers import PreTrainedTokenizerBase

def subject_pronoun_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Map for commonly observed pronouns likely to be subject pronouns or initiators
    pronouns = {"I", "you", "he", "she", "it", "we", "they", "one", "lily"}
    tokens = [tokenizer.decode(tok) for tok in toks.input_ids[0]]

    # Find the subject pronoun and place high attention scores to adjacent words
    for i, token in enumerate(tokens):
        if any(tok.lower() in token.lower() for tok in pronouns):
            out[i, i] = 1.0  # high attention to self
            if i + 1 < len_seq:
                out[i, i + 1] = 0.98  # next token
            if i + 2 < len_seq:
                out[i, i + 2] = 0.97  # next of next token

    # Ensure no row is all zeros by adding minimal attention to the last token (often EOS)
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Subject Pronoun Attention Pattern", out