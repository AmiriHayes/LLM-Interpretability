def rare_word_dominance(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # A placeholder mechanism to weight tokens based on a pseudo-frequency estimate
    # Here, the assumption is that rarer words within their context get higher attention.
    # The toy example uses simple rules since real data is inaccessible.
    for token_idx in range(1, len_seq - 1):
        # Let's assume the weights are inversely proportional to the index
        # In reality, you'd access frequency data or an equivalent method
        out[token_idx, token_idx] = 1.0 / (token_idx + 1)

    out[0, 0] = 1  # Self-attention to CLS
    out[-1, 0] = 1  # EOS pattern recognization 

    # Normalize rows of the matrix as attention heads do in practice
    out = out / out.sum(axis=1, keepdims=True)
    return "Rare Word Dominance", out
