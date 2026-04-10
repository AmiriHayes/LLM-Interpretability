from transformers import PreTrainedTokenizerBase
import numpy as np

def conjunction_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # A possible index that identifies conjunctions
    # We need a method to provide token indexes
    # Here we assume the tokenizer outputs indices where conjunctions would be present
    conjunction_indices = [i for i, tok in enumerate(tokenizer.tokenize(sentence)) if tok.strip() in {"because", "and", "or", "so", "but"}]

    for idx in conjunction_indices:
        # Attention given by conjunction tokens
        out[idx, :] = 1 / len_seq  # Each conjunction attends distributedly to all tokens

    # cls and eos tokens attend themselves
    out[0, 0] = 1
    out[-1, -1] = 1

    # Ensure attention to sentence end if conjunction focus on zero row
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4 # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True) # Normalize
    return "Coordinating Conjunction Focus", out
