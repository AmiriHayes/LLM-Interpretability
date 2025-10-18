def main_subject_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    # Tokenize sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Find the first token in the sentence (usually the main subject)
    # Assuming the first content word after specials and determiners is the main subject
    token_ids = toks.input_ids[0].tolist()
    special_token_id = tokenizer.cls_token_id if tokenizer.cls_token_id else 0
    main_subject_index = 0

    # Skip special tokens
    for i, token_id in enumerate(token_ids):
        if token_id != special_token_id:
            main_subject_index = i
            break

    # Create a mapping for attention where the main subject token attends to most tokens
    # Normalize attention so that the main subject has the most focus
    for i in range(len_seq):
        if i == main_subject_index:
            out[main_subject_index, :] = 1
        else:
            out[i, main_subject_index] = 0.8

    # Set CLS and EOS token attention
    out[0, 0] = 1
    out[-1, -1] = 1

    # Normalize the output matrix by rows
    row_sums = out.sum(axis=1, keepdims=True)
    np.seterr(divide='ignore', invalid='ignore')  # Suppress warnings for division by zero
    out = np.nan_to_num(out / row_sums)

    return "Main Subject Attention", out
