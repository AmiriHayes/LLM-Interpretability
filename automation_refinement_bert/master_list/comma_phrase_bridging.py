def comma_phrase_bridging(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Find comma positions
    tokenized_sentence = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    comma_indices = [i for i, tok in enumerate(tokenized_sentence) if tok == ',']

    # Link commas to surrounding tokens to signify bridging
    for idx in comma_indices:
        if idx > 0:
            out[idx, idx - 1] = 1  # Previous token
        if idx < len_seq - 1:
            out[idx, idx + 1] = 1  # Following token

    # Normalize attention scores for good measure
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0
    out = out / out.sum(axis=1, keepdims=True)

    return "Comma-Phrase Bridging", out
