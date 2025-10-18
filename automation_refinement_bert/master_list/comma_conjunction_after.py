def comma_conjunction_after(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Simplify aligning of tokens vs words
    seq = sentence.split()
    token_to_word_mapping = {i: idx for idx, word in enumerate(seq) for i, subword in enumerate(tokenizer.tokenize(word))}

    # Loop through tokens except special tokens and build the pattern
    for i in range(1, len_seq - 1):
        # Check for verbs following commas, ignore first token as CLS
        if ',' in seq:
            comma_index = seq.index(',')
            if comma_index + 1 < len(seq):
                next_token_after_comma = comma_index + 1
                if token_to_word_mapping.get(i) == next_token_after_comma:
                    for j in range(comma_index + 1, len_seq - 1):
                        out[i, j] = 1

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4 # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True) # Normalize

    return "Comma Conjunction Followed by Verb Phrase", out
