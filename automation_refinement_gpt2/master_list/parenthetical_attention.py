def parenthetical_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    words = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    paren_indices = []

    # find indices for parenthetical commas or phrases
    for i, word in enumerate(words):
        if word in {',', '(', ')', '[SEP]', '[CLS]'} or word.endswith(','):  # use endswith to cover subword tokens ending in comma
            paren_indices.append(i)

    # Connect parenthetical commas with each other
    for i in paren_indices:
        for j in paren_indices:
            if i != j:
                out[i][j] = 1

    # Add attention from the start token and stop token to the rest
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return 'Parenthetical Phrase Attention', out
