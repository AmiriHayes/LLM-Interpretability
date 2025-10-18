def paired_element_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize the sentence to identify pairs in the pattern 'word1 , word2'
    words = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    words_str = ' '.join(words)

    # Search for patterns like 'word1 , word2' or 'word1 and word2'
    matches = re.finditer(r'([^,\s]+)\s*,\s*([^,\s]+)', words_str)
    and_matches = re.finditer(r'([^,\s]+)\s*and\s*([^,\s]+)', words_str)

    # Handle both 'word1 , word2' and 'word1 and word2' matches
    for match in matches:
        start_word, end_word = match.groups()
        if start_word in words and end_word in words:
            start_index = words.index(start_word)
            end_index = words.index(end_word)
            out[start_index, end_index + 1] = 1
            out[end_index + 1, start_index] = 1

    for match in and_matches:
        start_word, end_word = match.groups()
        if start_word in words and end_word in words:
            start_index = words.index(start_word)
            end_index = words.index(end_word)
            out[start_index, end_index + 1] = 1
            out[end_index + 1, start_index] = 1

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize attention matrix
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize

    return "Focus on Paired Word Elements", out
