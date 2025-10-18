def sentence_opening_salience(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Focus primarily on the first token in the sentence
    out[0, :] = 1  # The first token attends to all tokens
    out[:, 0] = 1  # All tokens attend to the first token

    # Ensure each row has some attention to the end token to avoid zero-sum conditions
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Add a small value to ensure no rows are zero entirely
    out = out / out.sum(axis=1, keepdims=True)  # Normalize rows

    return "Sentence Opening Salience", out
