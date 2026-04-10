import numpy as np
from transformers import PreTrainedTokenizerBase

def semantic_pivot_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Identify key semantic pivot positions
    # Hypothetical function to determine pivot positions. This should ideally
    # be based on some pre-processing to find important words. Here, manually choose nouns, verbs, etc.
    pivot_indices = [i for i in range(1, len_seq-1) if toks.input_ids[0][i].item() in {
        tokenizer.vocab.get('<s>', 0),
        tokenizer.vocab.get('</s>', 0),
        tokenizer.vocab.get('day', 0),
        tokenizer.vocab.get('knew', 0),
        tokenizer.vocab.get('wanted', 0),
        tokenizer.vocab.get('went', 0),
        tokenizer.vocab.get('share', 0),
        tokenizer.vocab.get('smiled', 0),
        tokenizer.vocab.get('together', 0),
        tokenizer.vocab.get('finished', 0),
        tokenizer.vocab.get('felt', 0),
    }]

    for pivot in pivot_indices:
        for j in range(len_seq):
            out[pivot, j] = 1
            out[j, pivot] = 1

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the attention matrix
    out += 1e-4 # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)    

    return "Core Semantic/Syntactic Pivot Words Attention", out