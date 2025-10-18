def semicolon_comma_pre_association(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    # Iterate over tokens to find semicolon and comma
    for i, token in enumerate(tokens):
        # Check if the token is a semicolon or comma
        if token == "," or token == ":":
            out[i, i] = 1  # Self allocation
            # Initialize search for associate tokens before the comma/semicolon
            j = i - 1
            while j >= 0:
                if tokens[j] == tokens[-1]:  # Stop before CLS or SEP token
                    break
                # Calculate the connection strength based on distance
                distance_strength = 1 / (i - j)
                out[i, j] = distance_strength
                j -= 1

    # Normalize the out matrix
    for row in range(len_seq):
        row_sum = out[row].sum()
        if row_sum == 0:
            out[row, -1] = 1.0  # Ensure no row is all zeros
        else:
            out[row] /= row_sum  # Normalize by sum

    return "Semicolon and Comma Pre-Association", out
