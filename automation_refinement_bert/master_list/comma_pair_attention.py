def comma_pair_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    encoded_sentence = tokenizer.tokenize(sentence)

    # Identify all comma positions
    comma_positions = [i for i, token in enumerate(encoded_sentence) if token == ',']

    # Apply attention to each pair of commas
    for i in range(len(comma_positions) - 1):
        start = comma_positions[i]
        end = comma_positions[i + 1]
        for j in range(start + 1, end):
            out[j][start] = 1
            out[j][end] = 1

    # Prevent any row from being all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row][-1] = 1.0

    return "Comma Pair Attention", out
