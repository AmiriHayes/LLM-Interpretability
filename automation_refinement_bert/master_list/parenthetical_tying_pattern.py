def parenthetical_tying_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Identifying positions of commas to simulate parenthetical patterns
    words = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    commas = [i for i, word in enumerate(words) if word == ',']

    # Link each comma to the other commas in the sequence, simulating a tie
    for i in commas:
        for j in range(len_seq):
            if j in commas:
                out[i, j] = 1
            elif j < len_seq - 1 and words[j+1] in {'.', '?', '!', '[SEP]'}:
                out[i, j+1] = 1

    # Ensure no row is left with all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Parenthetical Tying Pattern", out
