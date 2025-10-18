def last_token_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    for i in range(len_seq):
        out[i, -1] = 0.5
        out[i, -2] = 0.5
    return "Last Token Pattern", out
