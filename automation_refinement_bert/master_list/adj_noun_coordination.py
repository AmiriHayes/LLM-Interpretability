def adj_noun_coordination(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    adj_indices = []
    noun_indices = []

    # Assuming we have a list of adjectives and nouns to identify them in the token IDs.
    # These lists could be further refined based on linguistic datasets
    adjectives = {'good', 'heavy', 'silent', 'relentless', 'endless', 'complex', 'hopeful', 'old', 'quiet', 'calming'}
    nouns = {'horizon', 'afternoon', 'afternoon', 'sky', 'side', 'sentinel', 'afternoon', 'book', 'bag', 'year', 'point'}

    word_ids = toks.word_ids(0)  # Get the mapping between token and word indices

    # Implement a simple heuristic to find adjectives and nouns
    for idx, word in enumerate(sentence.split()):
        if word in adjectives:
            for token_idx in range(len_seq):
                if word_ids[token_idx] == idx:
                    adj_indices.append(token_idx)
                    break
        elif word in nouns:
            for token_idx in range(len_seq):
                if word_ids[token_idx] == idx:
                    noun_indices.append(token_idx)
                    break

    # Connecting adjectives with their associated nouns if they're adjacent
    for adj_idx in adj_indices:
        for noun_idx in noun_indices:
            if adj_idx + 1 == noun_idx or adj_idx == noun_idx + 1:
                out[adj_idx, noun_idx] = 1
                out[noun_idx, adj_idx] = 1

    # Ensure no row is all zeros by connecting tokens to the [SEP]
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Adjective-Noun Coordination Pattern", out
