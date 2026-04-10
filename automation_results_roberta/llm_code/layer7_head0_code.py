import numpy as np
from transformers import PreTrainedTokenizerBase

def boundary_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assign high attention to the [CLS] token to itself and [SEP] token
    out[0, 0] = 1  # [CLS] to [CLS] attention
    out[-1, 0] = 1  # [SEP] to [CLS] attention
    out[-1, -1] = 1  # [SEP] to [SEP] attention

    # Each token gets self-attention, especially punctuations at the boundary
    for i in range(1, len_seq - 1):
        if toks.input_ids[0][i].item() in [tokenizer.sep_token_id, tokenizer.cls_token_id, tokenizer.eos_token_id]:
            # Give attention to boundaries and initial punctuations like start of sentence
            out[i, 0] = 0.9  # High attention from initial tokens and boundary tokens to [CLS]
            out[i, -1] = 0.9  # High attention from initial tokens and boundary tokens to [SEP]
            out[i, i] = 1.0   # Self-attention

    # Ensure no row has all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize each row

    return "Sentence Boundary and Initial Token Attention", out