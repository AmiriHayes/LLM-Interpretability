def sentence_beginning_salience(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Generally gives higher attention/intensity to the first few tokens in the sentence.
    for i in range(1, len_seq - 1):
        distance_to_start = i / len_seq
        salience = max(0, 1 - distance_to_start)
        out[i, 0] = salience

    out[0, 0] = 1  # CLS token retains self-attention
    out[-1, 0] = 1  # EOS token retains self-attention

    # Normalize attention scores to sum to 1 across each row (excluding last row to mimic padding effects)
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Sentence Beginning Salience", out
