import numpy as np
from transformers import PreTrainedTokenizerBase

def sentence_start_objects(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    sentence_attention = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    # The first non-special token in each sentence attends heavily to it,
    # this is consistent with sentence initiators.
    primary_token_index = 1

    # Assign high attention to the first token in the main sentence.
    for i in range(1, len_seq):
        out[primary_token_index, i] = 1.0

    for row in range(len_seq): # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out = out / out.sum(axis=1, keepdims=True)  # Normalize

    return "Sentence Start Objects", out