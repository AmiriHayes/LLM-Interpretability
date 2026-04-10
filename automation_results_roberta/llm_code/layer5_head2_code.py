import numpy as np
from transformers import PreTrainedTokenizerBase

def shared_object_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> str:
    # Tokenize the input sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Find indices of specific tokens shared in contexts
    target_tokens = {"needle", "shirt"}

    # For each token in the tokenized output
    tokens = [tokenizer.decode([tid]) for tid in toks.input_ids[0]]
    target_indices = [i for i, token in enumerate(tokens) if token in target_tokens]

    # Focus attention on tokens that often are shared, such as 'needle' and 'shirt'
    for idx in target_indices:
        out[:, idx] = 1.0  # All tokens attend to target tokens

    # Normalize the matrix by row
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize

    return "Shared Object Attention", out
