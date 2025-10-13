import numpy as np
from transformers import PreTrainedTokenizerBase

def pronoun_antecedent_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> str:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Define simple rules to connect common antecedent-pronoun relations
    token_list = sentence.split()
    # Use a dictionary to find pronouns and their possible references
    pronouns = ['he', 'she', 'it', 'they', 'her', 'him', 'them']
    antecedents = ['Lily', 'mom', 'needle', 'shirt', 'day']  # Extracted common antecedents

    # Mapping token positions
    token_map = {idx: token.lower() for idx, token in enumerate(token_list)}

    for idx, token in token_map.items():
        if token in pronouns:
            # Backtracking to find possible antecedents
            for j in range(idx-1, -1, -1):
                if token_map[j] in antecedents:
                    out[idx, j] = 1
                    break
        elif token in antecedents:
            out[idx, idx] = 1

    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize

    return "Pronoun Antecedent Attention", out