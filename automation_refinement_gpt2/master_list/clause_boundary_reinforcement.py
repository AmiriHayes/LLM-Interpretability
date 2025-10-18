def clause_boundary_reinforcement(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Attention matrix focusing on initial tokens of each clause separated by common clause separators
    clause_separators = {',', '.', '!', '?', ';', '-', ':'}
    separators_indices = [i for i, tok in enumerate(toks.tokens()) if tok in clause_separators]
    separators_indices.append(len_seq - 1) # Ensure to consider EOS as a clause separator
    last_separator = 0
    for separator_index in separators_indices:
        for i in range(last_separator, separator_index + 1):
            out[i, last_separator] = 1  # Attend to the first token of the clause
        last_separator = separator_index + 1

    # Add slight self attention to maintain stability
    for i in range(len_seq):
        out[i,i] = 0.1

    # Handle CLS and EOS tokens if they exist
    out[0, 0] = 1  # CLS
    out[-1, -1] = 1 # EOS

    # Normalize
    out /= out.sum(axis=1, keepdims=True)

    return "Clause Boundary Reinforcement Pattern", out
