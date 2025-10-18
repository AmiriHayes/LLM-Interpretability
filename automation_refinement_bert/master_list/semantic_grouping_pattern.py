def semantic_grouping_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    words = sentence.split()
    # Define a simple semantic grouping based on adjacent contexts in sentence
    keywords = ['of', 'with', 'to', 'for', 'on', 'from', 'in', 'and'] 
    # Create a positional mapping between tokenizer and spaCy tokenization
    for i, token in enumerate(words):
        if token in keywords:
            if i + 1 < len(words):  # Attending to word immediately after keyword
                out[i, i + 1] = 1
            if i - 1 >= 0:  # Attending to word immediately before keyword
                out[i, i - 1] = 1
    for row in range(len_seq):
        # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0
    return "Semantic Grouping Based on Contextual Keywords", out
