def thematic_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize the sentence
    words = tokenizer.tokenize(sentence)

    # Find the main topics/themes in the sentence based on provided data
    sentence_themes = []
    for word in words:
        if any(char.isdigit() for char in word) or len(word) > 5:
            sentence_themes.append(word)

    # Add attention to sentence themes and their surrounding contexts
    for i, word in enumerate(words):
        if word in sentence_themes:
            for j in range(max(0, i-2), min(len_seq, i+3)):
                out[i, j] = 1.0

    # Ensure no row is left with all zeros by defaulting attention to [SEP] if necessary
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the attention matrix
    out += 1e-4  # To avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return 'Thematic Focus Pattern', out
