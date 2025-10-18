def sentence_boundary_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> (str, np.ndarray):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Attend strongly to [CLS], [SEP], and sentence-internal punctuation (., ?, !)
    for i in range(len_seq):
        tok_string = tokenizer.decode(toks.input_ids[0][i])
        if tok_string in ["[CLS]", "[SEP]", ".", ",", "?", "!"]:
            out[i, :] = 1  # Token attends to all tokens
            out[:, i] = 1  # All tokens attend to this token

    # Normalize attention
    out = out / out.sum(axis=1, keepdims=True)
    return "Sentence Boundary Attention", out
