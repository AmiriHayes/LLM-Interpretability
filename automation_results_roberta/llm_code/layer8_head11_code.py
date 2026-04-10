from transformers import PreTrainedTokenizerBase
import numpy as np

def sentence_boundary_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> str:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Assign high attention to <s> and </s>
    out[0, :] = 1  # High attention from <s> to all tokens
    out[:, 0] = 1  # High attention to <s> from all tokens
    # Normalize the out matrix to ensure attention sum to 1
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize rows
    return "Sentence Boundary Focus", out