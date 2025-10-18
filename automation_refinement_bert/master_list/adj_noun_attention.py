def adj_noun_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize with spaCy to access POS tags
    doc = nlp(sentence)
    word_to_token = {tok.text: i for i, tok in enumerate(doc)}

    for i, token in enumerate(doc):
        if token.pos_ == 'ADJ':  # If adjective
            # Find nominal dependent heads (nouns associated)
            for child in token.children:
                if child.pos_ == 'NOUN':
                    token_index = word_to_token[token.text]
                    child_index = word_to_token[child.text]
                    out[token_index+1, child_index+1] = 1
                    out[child_index+1, token_index+1] = 1

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the attention matrix
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Adjective-Noun Collocation Attention", out
