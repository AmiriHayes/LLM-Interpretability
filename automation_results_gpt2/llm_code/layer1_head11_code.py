import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase


def pronoun_core_topic_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Extract tokens
    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    pronouns = {'I', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'us', 'him', 'her', 'them', 'my', 'your', 'his', 'their'}
    important_terms = {'found', 'share', 'needle', 'happy', 'play', 'button', 'sharp', 'difficult'}

    # Iterate over tokens to form attention pattern
    for i, token in enumerate(tokens):
        # Special attention to pronouns and main subjects
        if token in pronouns:
            out[i, i] = 1.0  # Pronoun self-attention
            for j, other_token in enumerate(tokens):
                if i != j and other_token in important_terms:
                    out[i, j] = 1.0
        elif token in important_terms:
            out[i, i] = 1.0  # Important term self-attention

    # Ensure there is always attention to CLS and EOS
    out[0, 0] = 1.0  # CLS token attention
    out[-1, -1] = 1.0  # EOS token attention

    # Normalize so rows sum to 1 (softmax simulation)
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Pronoun and Core Topic Attention", out