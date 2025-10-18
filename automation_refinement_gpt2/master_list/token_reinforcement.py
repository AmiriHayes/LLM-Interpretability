def token_reinforcement(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Identify the main token to reinforce attention (this model shows inclination at the sentence start)
    # Anchors special tokens like the sentence start and end
    out[0, 0] = 1  # reinforce attention to the opening token

    # Reinforce each non-special token to start attention back to the first word of the sentence
    for i in range(1, len_seq - 1):
        out[i, 0] = 0.9

    # Add self-attention weightage
    for i in range(1, len_seq - 1):
        out[i, i] = 0.1

    out[len_seq - 1, 0] = 1  # End token attention back to start
    out += np.eye(len_seq) * 1e-5  # Small added value to ensure no zeros (optional)

    # Normalize rows for attention probabilistic pattern
    row_sums = out.sum(axis=1, keepdims=True)
    out /= row_sums

    return "Initial Token Reinforcement with Sentence Anchoring Pattern", out
