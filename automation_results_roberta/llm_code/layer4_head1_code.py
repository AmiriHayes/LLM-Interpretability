import numpy as np
from transformers import PreTrainedTokenizerBase

def sentence_boundary_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Set attention from start of sentence token to all tokens at sentence start and end
    out[0, :] = 1  # Universal attention from <s> to every token
    out[:, 0] = 1  # Universal attention to <s> from every token

    # Set attention specifically focusing on complete run-up of sentence from <s> and </s>
    out[:, -1] = 1  # Universal attention to </s> from every token
    out[-1, :] = 1  # Universal attention from </s> to every token

    out += 1e-4 # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True) # Normalize to distribute attention

    return "Sentence Boundary Attention Pattern", out