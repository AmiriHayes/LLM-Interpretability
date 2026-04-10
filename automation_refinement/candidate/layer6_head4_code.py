import numpy as np
from transformers import PreTrainedTokenizerBase

# Define a function to capture the Compound Component Association pattern

def compound_component_association(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize the sentence and acquire tokens, word IDs using tokenizer
    words = sentence.split()
    # Define mappings for token indices (input_ids) to word pieces (word_ids)
    word_id_map = toks.encodings[0].words
    word_piece_map = toks.encodings[0].tokens

    # Identify pairs with compounds (prefix or suffix components) based on the examples
    for i, tok_id in enumerate(toks.input_ids[0]):
        if i == 0 or i == len_seq - 1:
            continue  # Skip CLS and SEP
        word_piece = word_piece_map[i]
        if word_piece.startswith("##"):
            # Mark attention pattern at compound parts
            for j in range(1, len_seq):
                root_id = i - 1  # Consider the root of a compound word
                if word_id_map[j] == word_id_map[i]:
                    out[j, root_id] = 1
                    out[root_id, j] = 1

    out[0, 0] = 1  # CLS token
    out[-1, 0] = 1  # SEP token

    # Normalize attention matrix
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Compound Component Association", out