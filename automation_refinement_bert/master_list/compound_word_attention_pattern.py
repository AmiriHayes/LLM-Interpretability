def compound_word_attention_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    words = sentence.split()

    # Simple heuristic: treat sequences with two or more hash symbols as compound words
    for idx, token in enumerate(words):
        if '##' in token:
            # Find the starting index of the compound part
            compound_idx = idx - token.count('##')
            out[compound_idx, idx + 1] = 1  # shift by 1 to account for [CLS]
            out[idx + 1, compound_idx] = 1

    out[0, 0] = 1  # Attention to [CLS]
    out[-1, 0] = 1  # Attention to [SEP]

    # Normalize attention matrix to sum to 1 over each row
    out += 1e-4  # tiny value to avoid zero division errors
    out = out / out.sum(axis=1, keepdims=True)
    return 'Compound Word Attention', out
