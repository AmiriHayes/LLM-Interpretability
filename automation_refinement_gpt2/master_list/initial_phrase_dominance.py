def initial_phrase_dominance(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Hypothesis: Head focuses heavily on the initial phrase and distributed attention
    # Calculate attention weight based on the position in the sequence
    initial_weight = 1.0
    decay_factor = 0.9  # Decay factor for attention spread from the initial token

    # Assign attention based on the decay pattern starting from the first token
    for i in range(1, len_seq-1):
        out[0, i] = initial_weight * (decay_factor ** (i-1))
        out[i, 0] = initial_weight * (decay_factor ** (i-1))

    # Ensure CLS token at pos 0 and EOS token at -1 have self-attention
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize the attention matrix
    out = out / out.sum(axis=1, keepdims=True)
    return "Initial Phrase Dominance", out
