def initial_token_centralization(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assign maximum attention to the first token for simplicity,
    # adjusted for GPT-2 tokenizer alignment (consider CLS, EOS center tokens or padding as needed)
    for i in range(len_seq):
        out[i, 0] = 1  # All tokens pay strong attention to the first token, similar to an initial centralization 

    # Normalize the output matrix row-wise
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Initial Token Centralization", out
