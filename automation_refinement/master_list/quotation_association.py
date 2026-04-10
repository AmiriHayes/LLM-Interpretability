def quotation_association(sentence: str, tokenizer) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize sentence and detect quotation marks
    tokens = sentence.split()
    quote_indices = [i for i, token in enumerate(tokens) if "'" in token or '"' in token]
    quote_pairs = []

    # Pair the indices assuming quote-indexed pairs
    for i in range(0, len(quote_indices), 2):
        if i+1 < len(quote_indices):
            quote_pairs.append((quote_indices[i], quote_indices[i+1]))

    # Mapping tokens according to tokenizer
    word_to_tokens = []
    for word in tokens:
        current_token_id = len(word_to_tokens)
        for token in tokenizer.tokenize(word):
            word_to_tokens.append(current_token_id)

    # Apply quote attention
    for (q_start, q_end) in quote_pairs:
        tok_start = word_to_tokens[q_start]
        tok_end = word_to_tokens[q_end]
        out[tok_start + 1, tok_start + 1 : tok_end + 2] = 1
        out[tok_end + 1, tok_start + 1 : tok_end + 2] = 1

    # Mark [CLS] and [SEP]
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Quotation Association Pattern", out
