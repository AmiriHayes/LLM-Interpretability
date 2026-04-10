def temporal_sequence_linking(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Iterate over sentence tokens to establish temporal sequence links
    sentence_tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    significant_words = {'yesterday', 'last', 'before', 'after', 'today', 'now', 'tomorrow', 'next', 'first'}
    time_related_indices = []

    for i, token in enumerate(sentence_tokens):
        if token.lower() in significant_words:
            time_related_indices.append(i)

    # Create a pattern that simulates temporal linking
    for index in time_related_indices:
        for i in range(1, len_seq - 1):
            if i != index:
                out[i, index] += 1
                out[index, i] += 1

    out[0, 0] = 1  # [CLS] token
    out[-1, 0] = 1  # [SEP] token

    # Normalize the matrix by row
    row_sums = out.sum(axis=1, keepdims=True)
    np.divide(out, row_sums, where=row_sums != 0, out=out)

    return "Temporal Sequence Linking", out
