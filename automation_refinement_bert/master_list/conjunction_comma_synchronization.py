def conjunction_comma_synchronization(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Identify tokens in sentence
    encoded_tokens = toks.input_ids[0].cpu().numpy()
    # Create a regex pattern to identify elements like ', and'
    commas_indices = []
    conjunction_indices = []

    input_tokens = tokenizer.convert_ids_to_tokens(encoded_tokens)
    for i, token in enumerate(input_tokens):
        # Check for conjunctions and commas
        if token == ',' and i+1 < len_seq:
            if input_tokens[i+1] in ["and", "but", "or"]:
                commas_indices.append(i)
                conjunction_indices.append(i+1)

    # Apply attention pattern for each pair
    for comma_index, conjunction_index in zip(commas_indices, conjunction_indices):
        # symmetrical attention pattern
        out[comma_index, conjunction_index] = 1
        out[conjunction_index, comma_index] = 1

    # Ensure attention to [SEP] token
    for row in range(len_seq): 
        if out[row].sum() == 0 and row < len_seq-1: 
            out[row, -1] = 1.0

    # Normalize the matrix row by row
    out += 1e-4 # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Conjunction-Comma Synchronization", out
