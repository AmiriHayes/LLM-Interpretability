def parallel_structures_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    # Tokenize the input sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Split the sentence on punctuation to detect parallel structures.
    parallel_phrases = sentence.split(", ")

    # Process each parallel phrase
    for phrase in parallel_phrases:
        if " and " in phrase or " or " in phrase:
            # Detect indices of parallel elements
            words = phrase.split()
            idx_base = -1
            idx_list = []
            for idx, word in enumerate(words):
                if word in {"and", "or"}:
                    if idx_base != -1:
                        idx_list.append((idx_base, idx))
                    idx_base = idx + 1

            # Assign attention within parallel structures
            for idx_pair in idx_list:
                start, end = idx_pair
                for i in range(start, end):
                    for j in range(start, end):
                        if i != j:
                            out[i+1, j+1] = 1

    # Ensure every token has some attention by adding fallback attention to [SEP]
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, len_seq - 1] = 1.0

    # Normalize the output matrix
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize
    return "Parallel Structures Attention", out
