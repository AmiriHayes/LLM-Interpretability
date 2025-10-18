def morphological_affix_focus(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    for i in range(1, len_seq - 1):
        token_i = toks.input_ids[0][i].item()
        if token_i in tokenizer.convert_tokens_to_ids(["##s", "##y", "##est", "##ing", "##ly", "##er", "##ful", "##ment", "##tion", "##ity", "##ed", "##less"]):
            # If token has morphological affix, it attends strongly
            # to a candidate token (fairly arbitrary targetting)
            out[i, i - 1] = 0.3
            out[i, i + 1] = 0.3
            if i + 2 < len_seq:
                out[i, i + 2] = 0.1  # attend a bit further

    # Ensure all tokens have some attention, here to final token
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the matrix
    out += 1e-4  # Avoid division by zero
    out /= out.sum(axis=1, keepdims=True)  # Normalize by row

    return "Morphological Affix Focus Pattern", out
