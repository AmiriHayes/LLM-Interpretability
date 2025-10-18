def punctuation_distanced_association(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    words = sentence.split()
    doc = nlp(' '.join(words))

    # Map token indices between tokenizer and spaCy
    tok_to_spacy = {}
    for spacy_token in doc:
        span = tok_to_spacy.get(spacy_token.i, [])
        span.append(spacy_token.idx)
        tok_to_spacy[spacy_token.i] = span

    # Handle punctuation and associated tokens
    for i, token in enumerate(doc):
        if token.is_punct:  # Focus on punctuation
            for j in range(len(doc)):
                if abs(token.i - j) <= 1:  # Look for nearby tokens
                    out[i + 1, j + 1] = 1.0

    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize return matrix

    return 'Punctuation-Distanced Association', out
