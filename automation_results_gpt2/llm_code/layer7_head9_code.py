import numpy as np
from transformers import PreTrainedTokenizerBase

# Function to reflect the observed attention pattern of layer 7, head 9
def sentence_beginning_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> str:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Set attention strong at the first token (index 1 due to start as CLS and end as EOS)
    for j in range(1, len_seq - 1):
        out[1, j] = 1.0

    # Ensure no row is all zeros by putting some attention on the final token
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize
    out = out / out.sum(axis=1, keepdims=True)

    return "Sentence Beginning Pattern", out