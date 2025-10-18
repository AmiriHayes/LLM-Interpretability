def noun_verb_pairing(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenizing the sentence with spaCy
    doc = nlp(sentence)
    token_map = {}
    spacy_tokens = [token.text for token in doc]

    # Map spaCy tokens to Hugging Face token indices
    spacy_index = 0
    for idx in range(1, len(toks.input_ids[0]) - 1):
        hf_token = tokenizer.decode(toks.input_ids[0][idx])[0]
        while spacy_index < len(spacy_tokens) and spacy_tokens[spacy_index] not in hf_token:
            spacy_index += 1
        if spacy_index < len(spacy_tokens):
            token_map[spacy_index] = idx

    # Identify noun-verb pairs and update the attention matrix accordingly
    for token in doc:
        if token.dep_ in ('nsubj', 'nsubjpass','dobj') and token.head.pos_ == 'VERB':
            noun_index = token.i
            verb_index = token.head.i
            if noun_index in token_map and verb_index in token_map:
                out[token_map[noun_index], token_map[verb_index]] = 1
                out[token_map[verb_index], token_map[noun_index]] = 1

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the attention matrix
    out = out / (out.sum(axis=1, keepdims=True) + 1e-4)  # Avoid division by zero

    return "Noun-Verb Pairing", out
