import numpy as np

def parenthetical_disregard(sentence: str, tokenizer) -> tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.ones((len_seq, len_seq))

    # Token ids and the potential attention disregard index tracking
    word_ids = toks.word_ids()
    parenthetical_indices = set()
    active = False

    for i, token in enumerate(sentence.split()):
        if token in {',', ':', ';'}:
            active = not active
        if active:
            parenthetical_indices.add(i + 1)  # Plus one to account for CLS token

    # Ensuring no attention to parenthetical content
    for i in range(1, len_seq - 1):  # Skip CLS and SEP
        if i in parenthetical_indices:
            for j in range(1, len_seq - 1):
                out[i, j] = 0
                out[j, i] = 0

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Parenthetical Disregard", out