def special_token_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    special_tokens = tokenizer.all_special_ids
    for i in range(len_seq):
        if toks.input_ids[0][i] in special_tokens:
            for sp_tok in special_tokens:
                out[i, toks.input_ids[0] == sp_tok] = 1
        else:
            out[i, -1] = 1
    out = out / out.sum(axis=1, keepdims=True)
    return "Special Token Pattern", out
