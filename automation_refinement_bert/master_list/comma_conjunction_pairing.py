def comma_conjunction_pairing(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Identify the positions of commas and conjunctions (e.g., and, or, but)
    tokens = tokenizer.tokenize(sentence)
    conjunctions = {'and', 'or', 'but'}
    comma_indices = [i for i, tok in enumerate(tokens) if tok == ',']
    conjunction_indices = [i for i, tok in enumerate(tokens) if tok in conjunctions]

    # Pair commas with their closest subsequent conjunction
    for comma in comma_indices:
        for conj in conjunction_indices:
            if conj > comma:
                out[comma + 1, conj + 1] = 1  # Account for [CLS]
                break

    # Ensure each row has at least one non-zero entry by defaulting to [SEP] attention
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Comma Conjunction Pairing", out
