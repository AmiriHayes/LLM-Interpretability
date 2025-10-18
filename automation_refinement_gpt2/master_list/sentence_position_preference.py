def sentence_position_preference(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Hypothesis is that words earlier in the sentence receive more attention
    # than words later in the sentence
    for i in range(len_seq):
        inverse_position_weight = len_seq - i
        for j in range(len_seq):
            weight_multiplier = 1 if j < i else 0  # Preference for attention to previous tokens
            out[i, j] = inverse_position_weight * weight_multiplier

    # Normalize the attention matrix row-wise
    out += 1e-4
    out /= out.sum(axis=1, keepdims=True)

    # Make sure [CLS] and [SEP] tokens receive some base self-attention
    out[0, 0] = 1
    out[-1, 0] = 1

    return "Sentence Position Preference", out
