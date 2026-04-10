def sentence_structure_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Identify punctuation indices
    cls_index = 0
    sep_index = len_seq - 1
    punct_indices = [i for i, tok in enumerate(toks.tokens()[0]) if tok in [',', '.', ';', ':', '?']]

    # Allocate attention based on observed patterns
    for i in range(1, len_seq - 1):
        if i in punct_indices:
            # Punctuation pays attention to itself and the next token
            if i+1 < len_seq - 1:
                out[i, i + 1] = 1

        else:
            # Words pay attention to the next punctuation or end of sequence.
            next_punct = next((pi for pi in punct_indices if pi > i), sep_index)
            out[i, next_punct] = 1

    # Ensure [CLS] and [SEP] attend to themselves
    out[cls_index, cls_index] = 1
    out[sep_index, sep_index] = 1

    # Normalize attention matrix
    out /= out.sum(axis=1, keepdims=True)
    return "Sentence Structure with Emphasis on Punctuation", out
