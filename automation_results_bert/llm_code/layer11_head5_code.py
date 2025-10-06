from transformers import PreTrainedTokenizerBase
import numpy as np
from typing import Tuple

# Function implementing the hypothesis for Layer 11, Head 5's attention pattern
# Specifically captures attention on punctuation and sentence terminators.

def punctuation_and_sentence_end_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    attention_tokens = {".", "?", ",", "!", "'", "-"}

    token_list = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    # Assign higher attention weights to punctuation and sentence terminators
    for i, token in enumerate(token_list):
        if any(punc in token for punc in attention_tokens):
            # Emphasizing self-attention on tokens identified
            out[i, i] = 0.5
            if i < len_seq - 1:
                out[i, i + 1] = 0.5

    # Ensure that no row in the matrix is all zeros by augmenting column-wise softmax
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize

    return "Punctuation and Sentence End Pattern", out