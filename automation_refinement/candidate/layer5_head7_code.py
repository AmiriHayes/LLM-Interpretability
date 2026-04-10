import numpy as np
from transformers import PreTrainedTokenizerBase

def end_punctuation_specialization(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Attention towards the last token (usually is a [SEP] or end punctuations like dot, comma) 
    out[:, -1] = 1

    # Self-attention for special tokens [CLS], [SEP]
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize attention weights
    out = out / out.sum(axis=1, keepdims=True)
    return "End-Punctuation Specialization", out