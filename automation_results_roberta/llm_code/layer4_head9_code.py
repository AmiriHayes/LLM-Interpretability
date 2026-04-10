import numpy as np
from transformers import PreTrainedTokenizerBase

def emphasis_key_components(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    word_ids = [i for i, pair in enumerate(zip(toks.word_ids(batch_index=0), toks.input_ids[0])) if pair[0] is not None]
    special_ids = [0, len(word_ids) + 1]  # Corresponding to <s> and </s>

    # Assign self-attention to <s> and </s>
    out[special_ids[0], special_ids[0]] = 1.0
    out[special_ids[1], special_ids[1]] = 1.0

    # Assign emphasis on key components with slightly lesser attention on </s>
    for idx in range(1, len_seq - 1):
        if idx in word_ids:
            out[idx, special_ids[1]] = 0.7
            out[idx, idx] = 0.3

    # Normalize each row
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0
        else:
            out[row] /= out[row].sum()

    return "Emphasis on Key Semantic Components", out