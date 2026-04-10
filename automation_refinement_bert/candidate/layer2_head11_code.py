import numpy as np
from transformers import PreTrainedTokenizerBase

def pause_continuance_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> str:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Find pause and continuance by identifying punctuation like commas, colons, and continuation words like 'and'
    attention_triggers = {',', ';', 'and', 'but', ':', '-'}

    tokens = tokenizer.tokenize(sentence)
    punctuation_indices = [i for i, token in enumerate(tokens) if any(p in token for p in punctuation)]

    for i in punctuation_indices:
        # Spread attention to tokens just preceding and following punctuation
        if i > 0:
            out[i+1, i] += 1
        if i < len_seq - 1:
            out[i+1, i+2] += 1

    # Normalize output matrix by row
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4 # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)
    return 'Pause and Continuance Attention', out