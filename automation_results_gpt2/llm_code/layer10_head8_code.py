import numpy as np
from transformers import PreTrainedTokenizerBase

# Function to simulate the attention pattern observed in layer 10, head 8

def sentence_subject_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Simulate attention from the first significant token (assumed subject) to other tokens in sentence
    # Get first non-special token index as subject, ignoring [CLS], spaces etc.
    subject_index = None
    for i, token in enumerate(toks.input_ids[0]):
        # Heuristic: Find the first alphanumeric token as the subject index
        if token.item() not in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
            subject_index = i
            break

    if subject_index is not None:
        for i in range(len_seq):
            out[subject_index, i] = 1

    # Ensure no row in output matrix is all zeros, non-empty target to match attention
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Sentence Subject Focused Attention", out