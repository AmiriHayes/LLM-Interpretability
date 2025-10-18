def sentence_start_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Default rule: Strong emphasis on the sentence starting token.
    out[0, 0] = 1  # First token (CLS)

    for i in range(1, len_seq-1):
        out[i, 0] = 1  # Each token attends to the first token
        out[i, i] = 0.1  # Lesser self-attention bias

    out[-1, 0] = 1  # EOL token attention to the start
    out[-1, -1] = 0.1  # Lesser self-attention

    # Normalize out matrix by its rows to make the attention weights sum up to 1 per row
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Sentence Starting Token Emphasis", out
