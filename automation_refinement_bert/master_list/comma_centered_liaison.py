def comma_centered_liaison(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Iterate over tokens
    for i, token_id in enumerate(toks.input_ids[0]):
        # Check if the token is a comma (assuming token id for ',' is known, e.g., 1010)
        if token_id == 1010:  # Substitute with actual comma token ID
            # Create dependencies with preceding and succeeding phrases around commas
            if i > 0:
                out[i, i - 1] = 1
            if i < len_seq - 1:
                out[i, i + 1] = 1

    # Ensure no row is all zeros by giving attention to [SEP] token if necessary
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Comma-centered Liaison Pattern", out
