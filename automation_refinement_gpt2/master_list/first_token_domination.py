def first_token_domination(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # The first token (index 0) attention dominates over other tokens.
    for i in range(len_seq):
        if i != 0:
            out[i, 0] = 1.0
        else:
            out[i, i] = 1.0 # Self-attention for the first token

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out = out / out.sum(axis=1, keepdims=True)  # Normalize
    return "First Token Domination", out
