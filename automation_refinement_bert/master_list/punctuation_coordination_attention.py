def punctuation_coordination_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> np.ndarray:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize using spaCy for punctuation detection (if needed, e.g. if sentence is not tokenized enough)
    for i in range(1, len_seq):
        if toks.input_ids[0][i] == tokenizer.convert_tokens_to_ids(","):
            # Coordinate this comma with the subsequent punctuation tokens or coordinating conjunctions
            # Capture "commas to commas" connections
            for j in range(i + 1, len_seq):
                if toks.input_ids[0][j] == tokenizer.convert_tokens_to_ids(",") or \
                   toks.input_ids[0][j] == tokenizer.convert_tokens_to_ids(".") or \
                   "and" in tokenizer.convert_ids_to_tokens(toks.input_ids[0][j]):
                    out[i, j] = 1
                    out[j, i] = 1

    # Ensure each token has at least some sort of attention, to prevent isolated tokens
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Out += 1e-4 to avoid any zero rows affecting the normalization
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True) # Normalize

    return "Frequent Punctuation Coordination Attention", out
