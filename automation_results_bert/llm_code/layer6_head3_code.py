import numpy as np
from transformers import PreTrainedTokenizerBase

def emphasize_verbs_and_objects(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    words = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    # keywords to focus that are associated with action
    action_keywords = ["found", "difficult", "share", "went", "said", "sew", "smiled", "sharing", "fixing"]

    for i, word in enumerate(words):
        # Assuming the word is an action if it matches the keywords
        if any(kw in word for kw in action_keywords):
            for j in range(len_seq):
                if i != j:
                    out[i, j] = 1 / (abs(i - j) + 1)  # Closer tokens get more attention

    for row in range(len_seq):
        # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize by rows

    return "Emphasizing Action Verbs and Their Objects", out