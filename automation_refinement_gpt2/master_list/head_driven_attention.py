def head_driven_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Define a heuristic where high attention is given to the first token (head) of each clause.
    head_indices = [0]  # Start with the first token
    words = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    # Detect clause boundaries based on punctuations
    for i, word in enumerate(words):
        if any(punct in word for punct in [",", ".", ";", "!", "?"]):
            if i + 1 < len_seq:
                head_indices.append(i + 1)

    # Assign high attention to each head from the first clause head
    for head_idx in head_indices:
        for j in range(1, len_seq - 1):
            out[j, head_idx] = 1

    # Self-attention for CLS and EOS
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize attention scores by row
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Head-Driven Sentence Attention", out
