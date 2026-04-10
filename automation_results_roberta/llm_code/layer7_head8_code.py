import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def shared_object_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    words = sentence.split()
    # Iterate over sentence characters to map tokenizer offsets
    word_to_token = {}  # Dictionary to map word indices to token indices
    char_offset = 0

    for word_idx, word in enumerate(words):
        token = tokenizer(word, return_tensors='pt', add_special_tokens=False)
        num_tokens = token.input_ids.shape[1]
        for token_offset in range(num_tokens):
            word_to_token[word_idx + token_offset] = list(range(char_offset, char_offset + num_tokens))
        char_offset += num_tokens

    # Iterate through words and establish attention towards shared constructs
    for idx, word in enumerate(words):
        if 'share' in word or 'needle' in word:
            # Direct attention between shared words and the article before 'needle'
            if idx > 0:
                for t_i in word_to_token[idx]:
                    for t_j in word_to_token[idx - 1]:
                        out[t_i][t_j] = out[t_j][t_i] = 1
            for other_idx, other_word in enumerate(words):
                if other_idx != idx and ('share' in other_word or 'needle' in other_word):
                    for t_i in word_to_token[idx]:
                        for t_j in word_to_token[other_idx]:
                            out[t_i][t_j] = out[t_j][t_i] = 1

    # Ensure no row is all zeros by defaulting to end-of-sequence token attention
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the output attention weights
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize by row

    return "Shared Object Attention Pattern", out