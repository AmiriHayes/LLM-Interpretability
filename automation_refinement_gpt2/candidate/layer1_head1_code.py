python
import numpy as np
from transformers import PreTrainedTokenizerBase

def sentence_beginning_focus(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # From the examples, the first word seems to be receiving the highest attention for all tokens
    out[:, 0] = 1  # All tokens attend to the first token
    out[0, 0] = 1  # Self-attention for CLS token
    out[-1, 0] = 1 # Ensure EOS token also attends to the first token

    # Normalize the attention weights by row
    out = out / out.sum(axis=1, keepdims=True)

    return "Sentence Beginning Focus", out