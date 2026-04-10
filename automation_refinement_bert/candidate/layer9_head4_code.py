import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def sentence_boundary_association(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assign high attention between beginning of the sentence and sentence-ending punctuation
    # Identify the positions of comma, period, question mark, and exclamation mark
    sentence_endings = {
        tokenizer.encode(["."])[1]: 0.9,
        tokenizer.encode([","])[1]: 0.8,
        tokenizer.encode(["?"])[1]: 0.7,
        tokenizer.encode(["!"])[1]: 0.6,
    }
    for i in range(1, len_seq - 1):
        # Check if current token is a sentence boundary
        token_id = toks.input_ids[0][i].item()
        if token_id in sentence_endings:
            out[i, 0] = sentence_endings[token_id]  # attention to [CLS]
            out[i, -1] = sentence_endings[token_id]  # attention to [SEP]

    # Ensure other tokens have minimal attention to the start and end
    for row in range(1, len_seq - 1):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize
    out = out / out.sum(axis=1, keepdims=True)
    return "Sentence Boundary Association", out