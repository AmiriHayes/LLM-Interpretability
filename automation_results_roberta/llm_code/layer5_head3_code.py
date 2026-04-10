from transformers import PreTrainedTokenizerBase
import numpy as np

# Hypothesis: This head generally attends more strongly to the start of sentences, focusing attention heavily on the beginning token, especially the start-of-sequence token.
def sentence_start_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Assume that token zero is the <s> token, assign high attention to it.
    start_index = 0

    # Fill attention pattern: Each token attends significantly to the start token <s>
    for i in range(len_seq):
        out[i, start_index] = 1.0

    # Normalize out matrix by row so that attention weights sum up to 1
    out += 1e-4 # To avoid division by zero
    out = out / out.sum(axis=1, keepdims=True) # Normalize rows

    return "Sentence Start Attention", out
