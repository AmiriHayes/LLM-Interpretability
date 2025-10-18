def noun_modifier_collocation(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    words = sentence.split()
    doc = nlp(' '.join(words))
    # Create a mapping from spaCy token indices to tokenizer token indices
    align_dict = dict()
    index_in_sentence = 0
    for w_idx, wo in enumerate(words):
        current_tokens = tokenizer.tokenize(wo)
        num_tokens = len(current_tokens)
        for tok_idx, tok in enumerate(current_tokens):
            align_dict[index_in_sentence + tok_idx] = w_idx
        index_in_sentence += num_tokens

    for token in doc:
        if token.dep_ in {'amod', 'compound', 'nmod', 'det'}:
            head_index = align_dict[token.head.i]
            token_index = align_dict[token.i]
            out[token_index+1, head_index+1] = 1

    for row in range(len_seq):
        # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0
    return "Noun Modifier Collocation", out
