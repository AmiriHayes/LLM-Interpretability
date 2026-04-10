import numpy as np
from transformers import PreTrainedTokenizerBase

def sentence_boundary_coherence_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assign high attention to [CLS] and [SEP] tokens
    out[0, :] = 1
    out[:, 0] = 1
    out[-1, :] = 1
    out[:, -1] = 1

    # Focus on sentence coherence, enhancing attention to sentence-initial and -final (content tokens)
    for i in range(1, len_seq-1):
        out[i, 0] = 1  # Initial token gets [CLS] attention
        out[i, -1] = 1  # Final token gets [SEP] attention

    # Filling in diagonal ensures some level of internal token attention
    np.fill_diagonal(out[1:-1, 1:-1], 0.1)

    # Ensure normalization
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize
    return "Sentence Boundary and Coherence Pattern", out
