def initial_contextual_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assume tokenization is consistent and we can assume alignment of indices
    # Initialize the pattern
    for i in range(len_seq):
        for j in range(len_seq):
            # Implement the pattern of high attention from the first token to all others
            if i == 0:
                out[i, j] = 1
            # Implement the pattern of stem token having higher weight to the content words it dominates (ex: in 'The sun dipped below', sun would dominate 'dipped below')
            elif j == 0:
                out[i, j] = 0.1
            else:
                out[i, j] = 0

    # Normalize out matrix by rows
    row_sums = out.sum(axis=1, keepdims=True)
    for i in range(len_seq):
        if row_sums[i] == 0:
            out[i, -1] = 1
        else:
            out[i] /= row_sums[i]

    return "Initial Contextual Attention", out
