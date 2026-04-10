import numpy as np
from transformers import PreTrainedTokenizerBase
import re

def suffix_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Use regular expressions to find suffix patterns
    suffix_pattern = re.compile(r'(##s|##ly|##ing|##ed|##ness|##ity|##ment)')
    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    # Create attention for tokens with suffixes
    for i, token in enumerate(tokens[1:-1], start=1):
        if suffix_pattern.search(token):
            out[i] = 1 / (len_seq - 1)  # equally distribute attention across other tokens

    # self-attend on [CLS] and [SEP]
    out[0, 0] = 1
    out[-1, 0] = 1

    return "Suffix Attention Pattern", out