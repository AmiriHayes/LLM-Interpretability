def sentence_beginning_attention_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assign full attention to the first token
    out[0, 0] = 1
    for i in range(1, len_seq-1):
        out[i, 0] = 1
        out[0, i] = 1
    out[-1, 0] = 1

    # Normalize
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)
    return "Sentence Beginning Attention Pattern", out
