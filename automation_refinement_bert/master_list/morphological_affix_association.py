def morphological_affix_association(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Split sentence for morphological processing
    words = sentence.split()

    # Use a simple heuristic to detect affix morphology relations based on tokenized outputs
    for i, word in enumerate(words):
        token_ids = toks.word_ids(batch_index=0)
        token_positions = [idx for idx, id in enumerate(token_ids) if id == i]

        # Detect potential morphological association in position indexed tokens
        for pos1 in token_positions:
            for pos2 in token_positions:
                if abs(pos1 - pos2) > 0 and (toks.tokens(batch_index=0)[pos2].startswith("##") 
                                             or toks.tokens(batch_index=0)[pos1].startswith("##")):
                    out[pos1, pos2] = 1

    # Ensure no row is all zeros, set at least one attention towards SEP token
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the attention matrix
    out += 1e-4 # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True) # Normalize

    return "Morphological Affix Association", out
