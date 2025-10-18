def parenthetical_insertion(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Placeholder tokens for parenthetical phrases often include commas and parenthesis
    words = sentence.split()
    start = None
    pairings = []

    for i, word in enumerate(words):
        # If a parenthetical-like context starts
        if word.startswith('(') or word.startswith(','):
            if not start:
                start = i + 1
            continue

        # If a parenthetical-like context ends
        if word.endswith(')') or word.endswith(','):
            if start is not None:
                pairings.append((start, i + 1))
                start = None

    # Creating the matrix based on detected parenthetical associations
    for start, end in pairings:
        for i in range(start, end):
            out[0, i] = 1  # Attend mainly between separators and the first token

    # Add a small value to non-zero elements to ensure we don't end with sparse matrices
    out = np.clip(out, 1e-4, 1.0)
    out = out / out.sum(axis=1, keepdims=True) if out.sum(axis=1, keepdims=True).all() else out

    return "Parenthetical Insertion Association", out
