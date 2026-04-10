import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def contrastive_relationship_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Define heuristic:
    # Find tokens likely involved in contrastive conjunctions (e.g., but, yet, however) and highlight them
    words = sentence.split()
    contrastive_conjunctions = {'but', 'yet', 'however', 'although', 'though', 'still', 'nevertheless', 'despite'}

    # Create a dictionary aligning token positions
    token_alignment = {}
    current_pos = 0
    for i, word in enumerate(words):
        # Tokenize each word separately
        word_tokens = tokenizer.tokenize(word)
        for token in word_tokens:
            token_alignment[current_pos] = i
            current_pos += 1

    # Find positions of contrastive conjunctions
    for idx, word in enumerate(words):
        if word.lower() in contrastive_conjunctions:
            # Get corresponding token ids
            token_indices = [pos for pos, tok_idx in token_alignment.items() if tok_idx == idx]
            for pos in token_indices:
                if pos > 0:
                    # Connect current token with previous words likely affected, through contrastive relationships
                    out[pos, :pos] = 1
                # Self-attend
                out[pos, pos] = 1
                if pos < len_seq - 1:
                    out[pos, pos + 1:] = 1

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize matrix
    out += 1e-4 # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True) # Normalize

    return "Contrastive Relationship Attention", out
`
