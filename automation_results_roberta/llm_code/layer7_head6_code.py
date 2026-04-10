import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

# Function to identify attention around named entities and their coreferences.
def coreferential_pair_modeling(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Example heuristic: if a token is a named entity, it attends to pronouns or similar entities.
    # Let's assume named entities are identified or mapped manually in this example.
    named_entities = ["Lily", "mom", "needle", "shirt"]  # Example entities to consider

    # Mapping tokenizer IDs to actual tokens
    input_ids = toks.input_ids[0]
    token2word = {idx: tokenizer.decode([input_id]).strip() for idx, input_id in enumerate(input_ids)}

    # This function checks if a token is a named entity.
    def is_named_entity(token):
        return token.strip() in named_entities

    # This function checks if a token is a pronoun (for simplicity assume it checks some known pronouns).
    def is_pronoun(token):
        return token.lower() in {"she", "her", "it", "they", "them", "we"}

    # Define a simple rule: named entities (tokens that look like entities or pronouns) attend to specific other tokens.
    for i in range(1, len_seq - 1):  # exclude [CLS] and [SEP] periphery tokens
        current_token = token2word[i]
        if is_named_entity(current_token):
            # Assume co-referential entities and their pronouns get attention
            for j in range(len_seq):
                potential_referent = token2word[j]
                if is_named_entity(potential_referent) or is_pronoun(potential_referent):
                    out[i, j] = 1

    # Make sure at least one attention per token
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize attention across tokens
    out += 1e-4  # Avoid zero rows
    out = out / out.sum(axis=1, keepdims=True)

    return "Co-Referential Pair Modeling", out