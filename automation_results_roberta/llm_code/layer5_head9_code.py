import numpy as np
from transformers import PreTrainedTokenizerBase

def sentence_boundary_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # This head predominantly attends to the beginning of the sentence <s>
    for i in range(1, len_seq - 1):
        out[i, 0] = 1

    # Ensure the row for <s> itself attends to the last token or any cls conventionally
    out[0, len_seq - 1] = 1.0

    # Normally the end token attends to itself in many types of attention
    out[len_seq - 1, len_seq - 1] = 1.0

    return "Sentence Boundary Attention", out