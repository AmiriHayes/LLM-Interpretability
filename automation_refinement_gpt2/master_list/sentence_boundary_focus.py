def sentence_boundary_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Focus attention on sentence boundaries (first and last token)
    out[0, 0] = 1  # CLS-like self-attention at start
    out[-1, 0] = 1  # EOS-like self-attention at end

    for i in range(1, len_seq - 1):
        if i == len_seq - 2:
            # Penultimate token might focus more towards EOF
            out[i, i] = 0.5
            out[i, -1] = 0.5
        else:
            # All tokens somewhat uniformly concentrate on the start
            out[i, 0] = 1

    # Normalize to make sure each row sums to 1, mimicking probability distributions in attention
    row_sums = out.sum(axis=1, keepdims=True)
    out = out / row_sums

    return "Sentence Boundary Focus", out
