def initial_token_reference_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # High self-attention for the first token
    out[0, 0] = 1.0
    for i in range(1, len_seq):
        # Moderate to low attention from each token to the initial token
        out[i, 0] = 0.5
        # Higher attention from the initial token to itself
        out[i, i] = 0.1

    # Normalize attention weights per row
    out = out / out.sum(axis=1, keepdims=True)

    return "Initial Token Reference Attention Pattern", out
