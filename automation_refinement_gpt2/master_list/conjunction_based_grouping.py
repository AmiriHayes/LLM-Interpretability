def conjunction_based_grouping(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    doc = nlp(sentence)
    word_ids = {token.idx: i for i, token in enumerate(doc)}

    # Identifying conjunctions and their respective grouped nouns/phrases
    conjunction_indices = [i for i, token in enumerate(doc) if token.pos_ == 'CCONJ']
    for conj_index in conjunction_indices:
        left_tree = doc[conj_index].lefts  # Tokens on the left of the conjunction
        right_tree = doc[conj_index].rights  # Tokens on the right of the conjunction

        left_indices = [word_ids[token.idx] for token in left_tree if token.idx in word_ids]
        right_indices = [word_ids[token.idx] for token in right_tree if token.idx in word_ids]

        # Create links within each group and across the conjunction
        for i in left_indices:
            for j in right_indices:
                out[i+1, j+1] = 1
                out[j+1, i+1] = 1
        for i in left_indices:
            for j in left_indices:
                if i != j:
                    out[i+1, j+1] = 1
        for i in right_indices:
            for j in right_indices:
                if i != j:
                    out[i+1, j+1] = 1

    # Ensure [CLS] attends to itself and [SEP] to [CLS]
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalizing rows
    row_sums = out.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    out = out / row_sums

    return "Conjunction-Based Grouping", out
