def initial_token_dominance(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Predict attention primarily to the first token
    for i in range(len_seq):
        out[i, 0] = 1.0

    # Normalize attention to ensure each row sums to 1
    out = out / out.sum(axis=1, keepdims=True)
    return "Initial Token Dominance", out
