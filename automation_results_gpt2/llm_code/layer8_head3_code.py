import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def pronoun_proper_noun_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    token_ids = toks.input_ids.squeeze().tolist()
    # Assuming the first meaningful word is what holds the focus (ignoring special tokens)
    focus_index = 1  # Default in case no pronoun or proper noun is found
    for i, token_id in enumerate(token_ids):
        # Check if it's a pronoun or proper noun using the tokenizer vocabulary
        token_str = tokenizer.decode([token_id])
        if token_str.lower() in ["i", "you", "he", "she", "it", "we", "they"] or token_str.istitle():
            focus_index = i
            break
    # Set attention predominantly to the focused token
    out[:, focus_index] = 1.0
    # Normalize rows
    out /= out.sum(axis=1, keepdims=True)
    return "Pronoun or Proper Noun Focus Pattern", out