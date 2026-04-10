from transformers import PreTrainedTokenizerBase
import numpy as np

def semantic_object_identification(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # A function to identify possible noun phrases (simplified)
    def find_nouns_indices(tokens):
        noun_indices = []
        current = None
        for i, token in enumerate(tokens):
            if token.isupper() or token.endswith('ing') or token.endswith('ed'):
                if current is not None:
                    noun_indices.append((current, i))
                current = i
        if current is not None:
            noun_indices.append((current, len(tokens)))
        return noun_indices

    tokens = [token.replace('##', '') for token in tokenizer.convert_ids_to_tokens(toks.input_ids[0])]
    noun_indices = find_nouns_indices(tokens)

    # Assign attention scores to noun phrase positions (simplified rule)
    for start, end in noun_indices:
        for i in range(start, end):
            for j in range(start, end):
                out[i, j] = 1

    # Ensure CLS and SEP have self-attention and normalize
    out[0, 0] = 1
    out[-1, -1] = 1
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize attention by rows

    return "Semantic Role Labeling - Object Identification", out