from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def verb_noun_relationship_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Simplified verb-noun relationship identification using an adjacent word heuristic
    words = tokenizer.convert_ids_to_tokens(toks.input_ids[0], skip_special_tokens=False)
    doc = zip(range(len(words)), words)

    # Simulated logic to connect verbs with certain key nouns and key phrases frequently
    for i, word in doc:
        if word.lower() in ('to', 'for', 'of') and (i+1) < len(words):
            j = i + 1
            # Create tight association between the preposition and following noun/phrase
            out[i, j] = 1.0
            out[j, i] = 1.0

        if word.lower() in ('is', 'was', 'are', 'were', 'can', 'must') and (i+1) < len(words):
            j = i + 1 
            # Important verb-noun relationships: connect the verb to the next word(s) more strongly
            out[i, j] = 0.8

    # Ensure cls and sep tokens have some attention to not become isolated
    out[0, 0] = 1
    out[-1, -1] = 1
    # Adding a small amount of value to ensure no complete zero rows
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1e-4

    out = out + 1e-4  # Small regularization
    row_sums = out.sum(axis=1, keepdims=True)
    out = out / row_sums  # Normalize attention scores per token

    return "Verb-Noun Relationship Attention", out