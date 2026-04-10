import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def phrase_span_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Identify the first token representing a phrase (first main token or word in the sentence)
    first_token_index = 1  # GPT2 tokenizes with a prefix space, first relevant token starts from index 1

    # Distribute attention across the span until the next punctuation or coordinating conjunction
    coordination_words = {"and", "or", "but"}
    punctuation_tokens = {".", ",", "!", "?"}

    # Loop through tokens (assuming word corresponds to tokens until punctuation or coordination)
    for i in range(first_token_index, len_seq):
        word = sentence.split()[i-1]  # Map token index back to word index in sentence
        if word in coordination_words or any(p in word for p in punctuation_tokens):
            break
        out[first_token_index, i] = 1

    # Normalize the attention to sum to 1
    out += 1e-4  # Avoid division by zero during normalization
    out = out / out.sum(axis=1, keepdims=True)  # Normalize

    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, 0] = 1.0  # Attending to the first token as default behavior

    return "Phrase Span Attention Pattern", out