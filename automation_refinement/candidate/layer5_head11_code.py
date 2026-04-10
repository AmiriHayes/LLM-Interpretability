import numpy as np
from transformers import PreTrainedTokenizerBase

def volitional_strategy_mapping(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    word_attention = {'decide': 0.6, 'strategize': 0.6, 'plan': 0.5, 'intend': 0.5, 'will': 0.4}
    words = sentence.split()
    for idx, word in enumerate(words):
        if word in word_attention:
            importance = word_attention[word]
            for j in range(1, len_seq-1):
                if j != idx and words[j] != '[SEP]':
                    out[idx+1, j] = importance
    for i in range(len_seq):
        out[i, 0] = 1  # Start with [CLS] token
    out[0, 0] = 1
    out[-1, 0] = 1
    out = out / out.sum(axis=1, keepdims=True)  # Normalize
    return "Volitional Strategy Mapping", out