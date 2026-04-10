def relative_position_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assign relative position importance to special tokens [CLS] and [SEP]
    out[0, :] = 1   # [CLS] self-attention
    out[:, 0] = 1   # [CLS] attends to all

    out[-1, :] = 1  # [SEP] self-attention
    out[:, -1] = 1  # [SEP] attends to all

    # Calculate relative distance decay, favoring 'central' tokens
    center = len_seq // 2
    for i in range(1, len_seq-1):
        dist_from_center = abs(center - i)
        decayed_importance = 1 / (1 + dist_from_center)
        out[i, :] += decayed_importance

    # Normalize out matrix by row to simulate attention distribution
    out = out / out.sum(axis=1, keepdims=True)
    return "Relative Position Attention", out
