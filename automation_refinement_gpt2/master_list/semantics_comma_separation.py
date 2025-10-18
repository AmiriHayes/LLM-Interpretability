def semantics_comma_separation(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    cls_idx = 0
    sep_idx = len_seq - 1
    # Assign strong self-attention to CLS and SEP tokens
    out[cls_idx, cls_idx] = 1
    out[sep_idx, sep_idx] = 1
    # Split the sentence into segments between commas
    sentence_segments = sentence.split(",")
    start = 0
    for segment in sentence_segments:
        segment_tokens = tokenizer(segment, return_tensors="pt")
        # Assign high attention weight within each segment
        for i in range(start + 1, start + len(segment_tokens.input_ids[0])):
            out[cls_idx, i] = 1
            out[i, cls_idx] = 1
            for j in range(start + 1, start + len(segment_tokens.input_ids[0])):
                out[i, j] = 1
        start += len(segment_tokens.input_ids[0])
    # Normalize the attention matrix by rows
    out /= out.sum(axis=1, keepdims=True)
    return "Semantics-Driven Comma Separation", out
