def compound_element_association(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Split sentence into words assuming tokenizer gives word_ids
    word_tokens = toks.word_ids(batch_index=0)
    word_to_tokens = {}

    # Map each word to its corresponding tokens
    for idx, word_id in enumerate(word_tokens):
        if word_id is None:
            continue
        if word_id not in word_to_tokens:
            word_to_tokens[word_id] = []
        word_to_tokens[word_id].append(idx)

    # Identifying compound elements within the tokens
    # Look for pattern where subparts are related within compounds
    for word_id, token_indices in word_to_tokens.items():
        if len(token_indices) > 1:  # Indicates a compound element
            for token_i in token_indices:
                for token_j in token_indices:
                    if token_i != token_j:
                        out[token_i, token_j] = 1

    # Normalize attention
    row_sums = out.sum(axis=1, keepdims=True)
    np.divide(out, row_sums, out=np.zeros_like(out, dtype=float), where=row_sums!=0)

    # Including [CLS] and [SEP] importance
    out[0, 0] = 1
    out[-1, 0] = 1

    return "Compound Element Association", out
