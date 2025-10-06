import numpy as np
from transformers import PreTrainedTokenizerBase

# Function to model the attention pattern for coordinating conjunctions and verbs
# Ignores starter/ender tokens for simplicity

def coordination_and_dependency(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    coord_conjs = {'and', 'but', 'or', 'so', 'yet'}
    verbs = {"'m", "'re", "'s", 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'am', 'have', 'has', 'had', 'do', 'does', 'did'}

    # Iterate through tokens to identify conjunctions and verbs
    for i, token in enumerate(tokens):
        if token in coord_conjs or token.endswith('ing'):
            # Look for potential dependencies on token to the right or left
            for j in range(max(i-3, 0), min(i+4, len_seq)):
                if j != i and tokens[j].lower() in verbs:
                    # Fill attention between current token and verbs
                    out[i,j] = 1
                    out[j,i] = 1
        elif token.lower() in verbs:
            # Reflectively fill across coordinating conjunctions
            for j in range(max(i-3,0),min(i+4,len_seq)):
                if j != i and tokens[j] in coord_conjs:
                    out[i,j] = 1
                    out[j,i] = 1

    # Normalize
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, 0] = 1.0 # default to CLS
        else:
            out[row] += 1e-4  # Avoid zero-feedback
            out[row] /= out[row].sum()  # Normalize

    return "Coordination and Dependency Parsing Function", out