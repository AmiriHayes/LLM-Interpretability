from transformers import PreTrainedTokenizerBase
import numpy as np
from typing import Tuple


def code_structure_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Splitting the sentence to identify key code structure tokens
    words = sentence.split()
    keywords = {"def", "for", "if", "while", "return", "import"}
    keyword_indices = [i for i, word in enumerate(words) if any(word.startswith(k) for k in keywords)]

    # Self-attention for keywords
    for idx in keyword_indices:
        out[idx+1, idx+1] = 1

    # Add attention from line start tokens to other line start tokens
    for i in range(len(words)):
        if any(words[i].startswith(k) for k in keywords) and i+1 < len_seq:
            out[i+1, :] += 0.1  # This value can represent a weaker attention compared to self-attention

    # CLS token self-attention
    out[0, 0] = 1

    # EOS token self-attention
    out[-1, -1] = 1

    # Normalize attention matrix by row to mimic softmax applied in transformers
    out += 1e-4  # avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Code Structure Attention Pattern", out