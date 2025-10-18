def initial_token_anchoring(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Set strong attention from all tokens to the initial token
    for i in range(1, len_seq-1):
        out[i, 0] = 1
    # Ensure the [CLS] and [SEP] have self-attention
    out[0, 0] = 1
    out[-1, 0] = 1
    out += 1e-4  # Regularization to prevent complete sparsity
    out /= out.sum(axis=1, keepdims=True)  # Normalize attention
    return "Initial Token Anchoring", out
