def cooccurrence_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    # Tokenize the input sentence
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Attention rule based on frequent cooccurrences
    # We focus on prepositions and their direct objects within a sentence.
    doc = sentence.split()
    preposition_indices = []
    prepositions = {'in', 'on', 'with', 'of', 'to'}  # Common prepositions

    # Identify indices of prepositions and associated objects in the tokenized sequence
    for i, word in enumerate(doc):
        if word in prepositions:
            preposition_indices.append(i)

    # Assign higher attention weights to specific word linkages
    for preposition_index in preposition_indices:
        for j in range(preposition_index + 1, len(doc)):
            if doc[j] not in prepositions:
                out[preposition_index + 1, j + 1] = 1
                out[j + 1, preposition_index + 1] = 1
                break

    # Normalize out matrix rows (attention distribution)
    for row in range(len_seq): 
        if out[row].sum() == 0:
            out[row, -1] = 1.0
        else:
            out[row] = out[row] / out[row].sum()  # Normalize to ensure sum is 1

    return 'Cooccurrence Attention Pattern', out
