import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')

def sentence_boundary_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assign CLS token attention
    out[0, 0] = 1.0  # <s> attends to itself primarily

    # E.g., focusing on attention towards punctuation like periods as indicators of sentence boundaries
    for i in range(1, len_seq-1):
        if tokenizer.decode(toks.input_ids[0][i]).strip() in {'.', '?', '!'}:
            out[i, 0] = 0.5  # Attention from periods (?, !)
            out[0, i] = 0.5  # Attention to periods (?, !) from <s>

    # Ensure each row in out matrix sums to 1 for proper attention distribution
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize

    for row in range(len_seq):
        if out[row].sum() == 0:  # Guarantee no row is all zeros
            out[row, -1] = 1.0

    return "End-of-Sentence and Start-of-Sentence Structure Recognition", out