def sentence_level_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # For simplicity, assume the sentence starts with a common structural token and mostly attends there
    # rather than heavily attending to specific content tokens.
    main_attention_token_idx = 0  # Assume CLS-like attention to the first available token
    secondary_attention_token_idx = len_seq - 1  # Including attention to the last token

    # Apply sentence-level attention distribution
    for i in range(len_seq):
        if i == 0:
            out[i, i] = 1
        elif i == main_attention_token_idx:
            out[i, main_attention_token_idx] = 0.6
            out[i, secondary_attention_token_idx] = 0.4
        elif i == secondary_attention_token_idx:
            out[i, main_attention_token_idx] = 0.4
            out[i, secondary_attention_token_idx] = 0.6
        else:
            out[i, main_attention_token_idx] = 0.9
            out[i, secondary_attention_token_idx] = 0.1

    # Very basic simulation of normalization
    out += 1e-5  # Avoid division by zero in some implementations
    out /= out.sum(axis=1, keepdims=True)

    return "Sentence-Level Attention Center", out
