import numpy as np
from transformers import PreTrainedTokenizerBase

def special_token_attention(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # <s> is assumed to be at position 0 after tokenization
    sos_token_index = 0  # This corresponds to <s> in the RoBERTa model
    eos_token_index = len_seq - 1  # Assume eos or sentence-end token is at the last index

    # Highly weight attention towards special tokens <s> and . (end of sentence)
    for token_index in range(1, len_seq - 1):
        # Attention towards <s>
        out[token_index, sos_token_index] = 1.0

    # Ensure no row is all zeros by assigning some attention to the EOS token
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, eos_token_index] = 1.0

    # Normalize the attention scores so that each row sums to 1
    out += 1e-4  # Avoid any division by zero errors
    out = out / out.sum(axis=1, keepdims=True)

    return "Special Token Attention Pattern", out