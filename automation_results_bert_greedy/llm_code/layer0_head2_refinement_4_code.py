import numpy as np
from transformers import PreTrainedTokenizerBase

def compound_word_affix_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    out[0, 0] = 1  # [CLS] attends to itself
    out[-1, -1] = 1 # [SEP] attends to itself
    # We check for tokens with affix markers (e.g., '##'), which are part of compound split words
    for i in range(1, len_seq - 1):
        if toks.input_ids[0][i].item() == tokenizer.convert_tokens_to_ids('##p'):
            out[i, i - 1] = 1  # current affix attends to the prior token which is its root
        elif toks.input_ids[0][i].item() == tokenizer.convert_tokens_to_ids('##w'):
            out[i, i - 1] = 1
    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0
    return "Compound Word Affix Focus", out