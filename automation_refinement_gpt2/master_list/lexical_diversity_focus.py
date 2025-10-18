def lexical_diversity_focus(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize the sentence with tokenizer and identify unique tokens
    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    unique_tokens = list(set(tokens))

    # Map each unique token its occurrences in the sentence
    token_map = {token: [] for token in unique_tokens}
    for i, token in enumerate(tokens):
        if token in token_map:
            token_map[token].append(i)

    # Assign high attention to unique tokens and medium to others
    for token, indices in token_map.items():
        for idx in indices:
            if len(indices) == 1:  # Unique token
                out[idx, idx] = 1
            else:  # Non-unique token
                attention_value = 0.5
                for i_idx in indices:
                    out[i_idx, idx] = attention_value

    # Assign attention to [CLS] and [EOS] tokens
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize the matrix
    out = out / out.sum(axis=1, keepdims=True)
    return "Lexical Diversity Focus and Attention Pattern", out
