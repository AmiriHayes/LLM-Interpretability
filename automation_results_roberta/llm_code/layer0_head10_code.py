from transformers import PreTrainedTokenizerBase
import numpy as np

def sentence_beginning_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Focus maximum attention on the sentence start token <s>
    for i in range(1, len_seq-1):
        out[i, 0] = 1

    # Ensure the special tokens <s> and </s> have some attention
    out[0, 0] = 1  # Self-attention for <s>
    out[-1, 0] = 1  # Attention from </s> to <s>

    # Normalize attention weights
    out = out / out.sum(axis=1, keepdims=True)

    return "Sentence Beginning Pattern", out