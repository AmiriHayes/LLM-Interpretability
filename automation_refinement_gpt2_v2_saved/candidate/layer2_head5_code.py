import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

# Function to simulate attention behavior of Layer 2, Head 5 in GPT2LMHeadModel
# Based on the hypothesis that this head is tracking function definitions

def function_definition_influence(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    token_to_word = toks.word_ids()  # Word-aligned token ids
    def_indices = []

    # Locate positions of 'def' and similarly influence across lines for cohesive code blocks
    for i, token in enumerate(toks.input_ids[0]):
        word_id = token_to_word[i]
        if word_id is not None and 'def' in toks.decode([token]):
            def_indices.append(i)

    # Influence pattern for 'def' tokens
    for index in def_indices:
        # Create a block of attention for tokens contributing to function definition
        # Expand some contextual attention just below the word 'def', it spreads upwards
        min_context = max(0, index - 5)
        max_context = min(len_seq, index + 5)

        for sub_index in range(min_context, max_context):
            out[index, sub_index] = 1  # Attention emanates from the 'def' root
            out[sub_index, index] = 1

    # Normalize the matrix
    out[0, 0] = 1
    out[-1, 0] = 1
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Function Definition Influence", out