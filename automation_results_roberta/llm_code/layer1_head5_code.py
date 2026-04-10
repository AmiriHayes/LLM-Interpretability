import numpy as np
from transformers import PreTrainedTokenizerBase

def sentence_boundary_detection(sentence: str, tokenizer: PreTrainedTokenizerBase) -> str:
    # Tokenize the input sentence
    toks = tokenizer([sentence], return_tensors="pt")
    # Length of sequence including <s> and </s> tokens
    len_seq = len(toks.input_ids[0])
    # Initialize output matrix
    out = np.zeros((len_seq, len_seq))

    # Attention to beginning of sentence <s>
    for i in range(1, len_seq-1):
        out[i, 0] = 1

    # Ensure that the CLS and SEP tokens attend to themselves
    out[0, 0] = 1
    out[-1, -1] = 1

    # Normalize the attention matrix so each row sums to 1
    out = out / out.sum(axis=1, keepdims=True)

    return "Sentence Boundary Detection", out