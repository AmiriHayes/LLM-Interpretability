def prepositional_phrase_association(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Use spaCy to parse the sentence
    doc = nlp(sentence)
    token_alignments = {token.idx: i for i, token in enumerate(doc) if i < len_seq}

    # Loop through spaCy tokens to find prepositional phrases
    for token in doc:
        if token.dep_ == 'prep':
            prep_index = token_alignments.get(token.idx, -1)
            # Connect preposition to its object and directly related tokens
            for child in token.children:
                child_index = token_alignments.get(child.idx, -1)
                if child_index != -1:
                    out[prep_index, child_index] = 1
                    out[child_index, prep_index] = 1

    # Ensuring no row in the output matrix is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # Defaults to SEP token

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize

    return "Prepositional Phrase Association", out
