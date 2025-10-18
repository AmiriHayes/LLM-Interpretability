def punctuation_pair_association(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    # Tokenize the sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    punctuation_marks = {',', '.', '!', '?', ':', ';', '-', '(', ')', '[', ']', '{', '}'}
    punctuation_indices = [i for i, token in enumerate(tokens) if token in punctuation_marks]

    # Loop through each pair of punctuation marks
    for i in range(len(punctuation_indices) - 1):
        idx1 = punctuation_indices[i]
        idx2 = punctuation_indices[i + 1]

        # Link the two punctuation tokens
        out[idx1, idx2] = 1
        out[idx2, idx1] = 1

    # Ensure no row is all zeros by setting the last column to 1
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the attention weights
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Punctuation Pair Association", out
