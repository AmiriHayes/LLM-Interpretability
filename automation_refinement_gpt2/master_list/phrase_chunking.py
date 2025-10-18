def phrase_chunking(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    # Tokenize sentence
    toks = tokenizer([sentence], return_tensors='pt')
    token_ids = toks.input_ids[0]
    len_seq = len(token_ids)

    # Initialize attention matrix
    out = np.zeros((len_seq, len_seq))

    # Identify segment bounds based on common punctuation marks and standard phrase boundaries
    chunk_boundaries = {i for i, tok in enumerate(token_ids) if tok in tokenizer.convert_tokens_to_ids([',', '.', ':', ';', '?', '!', '"'])}
    chunk_boundaries.update({0, len_seq - 1})  # include start and end of the sentence

    # Assign attention to each token within its determined chunk
    for i in range(len_seq):
        for j in range(i, -1, -1):  # look backward
            if j in chunk_boundaries:
                break
            out[i, j] = 1
        for k in range(i, len_seq):  # look forward
            if k in chunk_boundaries:
                break
            out[i, k] = 1

    # Normalize attention matrix
    out += 1e-4  # avoid division by zero issues during normalization
    out = out / out.sum(axis=1, keepdims=True)

    return "Phrase Chunking Pattern", out
