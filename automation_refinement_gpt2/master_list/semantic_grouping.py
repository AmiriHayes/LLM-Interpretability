def semantic_grouping(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    attention_dict = {}

    # Hypothesis is that certain content words like nouns or subjects are attracting initial tokens
    for idx, token in enumerate(tokens):
        out[idx, 0] = 1
        if token not in attention_dict:
            attention_dict[token] = [idx]
        else:
            attention_dict[token].append(idx)

    # Map first non-punctuation token to each token in its respective semantic group
    for token, indices in attention_dict.items():
        if len(indices) > 1:
            for idx in indices:
                for j in indices:
                    out[idx, j] = 1
                    out[j, idx] = 1

    # CLS and SEP tokens receive their own distinct attention, marked at the [0][0] and last[0] index.
    out[0, 0] = 1
    out[-1, 0] = 1
    # Normalize the attention distribution by row to have a uniform attention weight sum
    out = out / out.sum(axis=1, keepdims=True)
    return "Semantic Grouping Pattern", out
