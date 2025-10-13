from transformers import PreTrainedTokenizerBase
import numpy as np

# Hypothesis: The head focuses its attention primarily on the beginning tokens of the sentence.
def beginning_attention(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Set high attention for the first token across all other tokens
    for i in range(len_seq):
        out[i, 0] = 1  # Strong attention to the first token (begin of sentence)

    # Normalize the attention matrix
    out /= out.sum(axis=1, keepdims=True)
    return "Sentence Beginning Attention Pattern", out