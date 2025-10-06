import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple


def conjunction_influence(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Define typical conjunction indices; these tokens often include words like 'and', 'but', 'because', etc.
    conjunction_words = {'and', 'but', 'because', 'so'}
    word_ids = toks.word_ids(batch_index=0)
    word_to_idx = {word: idx for idx, word in enumerate(word_ids) if word is not None}

    # Map tokens to their positions in the input
    for i, j in enumerate(word_ids):
        if j is None:
            continue
        if tokenizer.decode(toks.input_ids[0, i]).strip().lower() in conjunction_words:
            for word, idx in word_to_idx.items():
                if idx > i:
                    out[i, idx] = 1

    # Normalize attention
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0
        else:
            out[row] = out[row] / out[row].sum()
    return "Conjunction Influence Pattern", out