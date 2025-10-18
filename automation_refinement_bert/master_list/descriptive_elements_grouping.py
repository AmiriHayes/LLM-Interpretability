def descriptive_elements_grouping(sentence: str, tokenizer: PreTrainedTokenizerBase) -> str:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize the sentence using spaCy to find POS tags
    doc = nlp(sentence)
    # Align spaCy tokens and tokenizer output
    spacy_tokens = [token.text for token in doc]
    token_to_spacy = {i: o for i, o in enumerate(doc) if o.text in str(toks)}

    # Function to detect descriptive elements and their associations
    def is_descriptive(token):
        return token.pos_ in ['ADJ', 'ADV']

    for i, token in token_to_spacy.items():
        if is_descriptive(token):
            for j, related_token in enumerate(doc):
                if related_token in token.children or token in related_token.children:
                    # Presuming a child relationship indicates descriptive context connection
                    out[i + 1, j + 1] = 1

    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # Ensure no row is all zeros

    out += 1e-4  # Slight adjustment to avoid purely zero rows and columns
    out = out / out.sum(axis=1, keepdims=True)  # Normalize

    return "Descriptive Elements Grouping Pattern", out
