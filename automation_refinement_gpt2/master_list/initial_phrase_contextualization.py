def initial_phrase_contextualization(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # The first token always attends to itself while all other tokens focus on the initial part of the sentence.
    attention_threshold = 0.7

    for i in range(1, len_seq):
        if i <= int(len_seq * attention_threshold):
            # Emphasizing strong attention to the first token among the first part tokens
            out[i, 0] = 1
        else:
            # Remaining tokens have reduced focus
            out[i, 0] = 0.5

    # First token has self-attention
    out[0, 0] = 1

    # Normalize by row
    out /= out.sum(axis=1, keepdims=True)
    return "Initial Phrase Contextualization", out
