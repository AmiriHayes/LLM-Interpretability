import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def narrative_structure_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Determine key narrative tokens based on examples, e.g., names, actions, and objects
    key_narrative_indices = {0}  # Always attend to [CLS]
    attention_triggers = ["One", "day", "Lily", "needle", "share", "mom", "Can", "Together", "After", "They"]

    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    for i, token in enumerate(tokens):
        if any(trigger.startswith(token) for trigger in attention_triggers):
            key_narrative_indices.add(i)

    # High attention to key narrative tokens
    for i in key_narrative_indices:
        out[i, list(key_narrative_indices)] = 1

    # Ensure no row is all zeros by allocating attention to [SEP] or end of sentence token
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize attention weights
    out += 1e-4  # Add small value to avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Attention on Key Elements of Narrative Structure", out
