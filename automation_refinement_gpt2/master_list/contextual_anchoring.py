def contextual_anchoring(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    # Tokenize the input sentence
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])  # Get the length of the sequence
    out = np.zeros((len_seq, len_seq))  # Initialize the attention matrix

    # Loop over each token, with the first token having the highest self-attention
    for i in range(1, len_seq):
        if i == 1:
            # The first word receives the strongest anchoring attention
            out[i, :] = 1.0  # Anchoring to the first token (CLS-like behavior without dominance)
        else:
            # The rest of the sentence's words receive progressively less attention
            # but are still rooted in the initial segment
            out[i, :i] = 1.0 / (i)

    # Since special tokens usually have fixed heads, let them attend to themselves
    out[0, 0] = 1
    out[-1, -1] = 1 

    # Normalize to mimic attention weights
    out /= out.sum(axis=1, keepdims=True)

    # Return the identified pattern name and the built attention matrix
    return 'Sentence-Initiated Contextual Anchoring', out
