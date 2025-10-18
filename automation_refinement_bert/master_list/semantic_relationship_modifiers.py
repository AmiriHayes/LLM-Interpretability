def semantic_relationship_modifiers(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    word_mappings = {
        "of": "modifier",
        "for": "modifier",
        "with": "modifier",
        "on": "modifier",
    }

    # Assume unique attention to contentful elements like adjectives or verbs and their targets
    for i, token in enumerate(sentence.split()):
        # Assign higher attention to target modifier words
        if token.lower() in word_mappings:
            out[i + 1, :] = 1 / (len_seq - 2)  # all except CLS and SEP

    # Ensure every token pays some attention even when not in pattern
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Avoid division by zero issues
    out = out / out.sum(axis=1, keepdims=True)  # Normalize

    return "Semantic Relationship Modifiers", out
