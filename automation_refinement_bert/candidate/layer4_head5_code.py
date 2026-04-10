import numpy as np
from transformers import PreTrainedTokenizerBase


def mathematical_equation_complexity_analysis(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenization often involves special tokens like 'x' and mathematical terms,
    # we assume that special mathematical equations of the form x^n or a*x,
    # or terms like `\frac` in LaTeX are given more attention.
    # Identify tokens here using the tokenizer or regex
    equation_tokens = [i for i, tok in enumerate(toks.input_ids[0]) if tok in [/* add token ids for 'x', '^', '\frac' etc. */]]

    for i in range(len_seq):
        if i in equation_tokens:
            for eq_index in equation_tokens:
                out[i, eq_index] = 1  # Highlight intra-equation attention

    # Normalize the attention pattern
    out[0, 0] = 1
    out[-1, 0] = 1
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Mathematical Equation Complexity Pattern", out

