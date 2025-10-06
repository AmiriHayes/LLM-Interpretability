import numpy as np
from transformers import BertTokenizer

def subordinate_conjunction_attention(sentence: str, tokenizer: BertTokenizer):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Token IDs for common subordinate conjunctions and coordinators:
    conjunctions = set(tokenizer.convert_tokens_to_ids(tok) for tok in [
        'because', 'if', 'when', ',', 'and', 'but', 'so'
    ])

    # Token IDs for commonly focused words before or after conjunctions:
    focus_words = set(tokenizer.convert_tokens_to_ids(tok) for tok in [
        'was', 'is', 'were', 'can', 'could'
    ])

    for i, token_id in enumerate(toks.input_ids[0]):
        # If token is a conjunction, emphasize attention on the next or previous token
        if token_id in conjunctions:
            if i + 1 < len_seq and toks.input_ids[0][i + 1] in focus_words:
                out[i, i + 1] = 1
            if i - 1 >= 0 and toks.input_ids[0][i - 1] in focus_words:
                out[i, i - 1] = 1

    # Normalizing the attention pattern
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0
        else:
            out[row] /= out[row].sum()

    return "Subordinate Clause and Coordinating Conjunction Emphasis", out