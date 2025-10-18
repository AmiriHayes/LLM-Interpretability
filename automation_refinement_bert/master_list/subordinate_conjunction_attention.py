def subordinate_conjunction_attention(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize the sentence with spaCy for linguistic analysis
    doc = nlp(sentence)

    # Align spaCy tokens with transformer tokens
    spacy_to_transformer = {i: j for i, token in enumerate(doc)
                            for j, tok_id in enumerate(toks.input_ids[0])
                            if (token.text in tokenizer.convert_ids_to_tokens([tok_id.item()]))}

    for token in doc:
        # Identify subordinate conjunctions and related words
        if token.dep_ == 'mark':
            conj_index = spacy_to_transformer[token.i]
            for ancestor in token.ancestors:
                ancestor_index = spacy_to_transformer.get(ancestor.i)
                if ancestor_index is not None:
                    out[conj_index, ancestor_index] = 1

    # Apply normalization to ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Avoid division by zero for normalization
    out = out / out.sum(axis=1, keepdims=True)

    return "Subordinate Conjunction Attention", out
