def initial_token_dominance(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Most significant attention comes from the first token
    out[0, :] = 1/(len_seq - 1) if len_seq > 1 else 0
    # Define non-zero attention over all tokens that are not the CLS token
    if len_seq > 1:
        out[0, 0] = 0
    # Apply normalization over columns
    if len_seq > 1:
        out[:, 0] = 1e-4
    out = out / out.sum(axis=1, keepdims=True)
    return 'Initial Token Dominance', out
