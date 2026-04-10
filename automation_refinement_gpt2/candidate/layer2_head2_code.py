import numpy as np
from transformers import PreTrainedTokenizerBase

def parenthetical_phrase_delimitation(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    word_ids = toks.word_ids()

    # Find brackets indices
    open_bracket_positions = [i for i, tok in enumerate(toks.tokens()) if tok == "("]
    close_bracket_positions = [i for i, tok in enumerate(toks.tokens()) if tok == ")"]

    # Assume a simplistic single pair for this hypothesis
    if open_bracket_positions and close_bracket_positions:
        open_bracket_pos = open_bracket_positions[0]
        close_bracket_pos = close_bracket_positions[0]

        # Set high attention between open and close brackets
        for i in range(open_bracket_pos+1, close_bracket_pos):
            out[open_bracket_pos, i] = 1.0
            out[i, close_bracket_pos] = 1.0

    # Normalize the attention to prevent zero rows
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Parenthetical Phrase Delimitation", out