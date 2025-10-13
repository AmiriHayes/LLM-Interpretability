import numpy as np
from transformers import PreTrainedTokenizerBase

def subject_centric_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    words = [word.strip() for word in sentence.split()]

    # Determine the heuristic 'subject' based on raw positional heuristics
    if '"' in words:
        words = [w for w in words if w != '"']
    subject_index = 1 if len(words) > 1 else 0 

    for i in range(1, len_seq-1):
        out[i, 1] = 1  # Assuming subject is the first content word (word 1 after special tokens)
        out[1, i] = 1

    out[0, 0] = 1  # CLS-like self attention for first token
    out[-1, -1] = 1  # EOS-like self attention for last token

    # Normalize rows
    out = out / out.sum(axis=1, keepdims=True)

    return "Subject-Centric Attention Pattern", out