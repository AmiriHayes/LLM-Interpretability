from transformers import PreTrainedTokenizerBase
import numpy as np

def sentence_boundary_attention(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # The highest attention scores seem to be on the special tokens <s> and </s> (start and end)
    for i in range(1, len_seq - 1):  # Avoid [CLS] and [SEP] (first and last tokens in most tokenizations)
        out[i, 0] = 0.5  # Assign half attention to the start token <s>
        out[i, -1] = 0.5  # Assign half attention to the end token </s>

    out[0, 0] = 1.0  # Ensure <s> (start token) has high self-attention
    out[-1, -1] = 1.0  # Ensure </s> (end token) has high self-attention

    # Normalize by row to make it a valid attention distribution (summing to 1 for each token)
    out += 1e-6  # Add a small constant to avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Sentence Start/End Position Focusing Pattern", out