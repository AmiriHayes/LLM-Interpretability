import numpy as np
from transformers import PreTrainedTokenizerBase

def initial_token_without_punctuation(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Tokenize the sentence for easier reference from the pattern
    toks_ids = toks.input_ids[0]
    toks_tokens = tokenizer.convert_ids_to_tokens(toks_ids)
    initial_word_idx = None
    # Find the index of the non-punctuation initial word
    for i, word in enumerate(toks_tokens):
        if word.isalnum():  # Check if the token is not punctuation
            initial_word_idx = i
            break
    if initial_word_idx is not None:
        for i, word in enumerate(toks_tokens):
            out[i, initial_word_idx] = 1
    # Handle [CLS] and [EOS] attention
    out[0, 0] = 1
    out[-1, 0] = 1
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)
    return "Initial Token Reference Ignoring Punctuation", out