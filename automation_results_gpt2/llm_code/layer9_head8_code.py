import numpy as np
from transformers import PreTrainedTokenizerBase

def sentence_start_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> (str, np.ndarray):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assuming attention primarily focuses on the first content token after any special token (like [CLS])
    first_word_idx = 1  # We generally skip the [CLS] or any special initial token
    # Assign high attention to the starting content token
    out[first_word_idx, :] = 1

    # Ensure self-attention for CLS and SEP/EOS tokens
    out[0, 0] = 1  # The [CLS] token
    out[-1, -1] = 1  # The [SEP] or [EOS] token

    # Normalize the attention matrix by row
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Sentence Starting Token Focus", out