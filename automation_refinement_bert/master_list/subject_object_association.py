def subject_object_association(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenization and alignment between tokenizer and spacy
    doc = nlp(sentence)
    align_dict = {}
    token_idx = 1  # Start at 1 due to [CLS]
    for token in doc:
        tok_span = tokenizer.encode(token.text, add_special_tokens=False)
        for _ in tok_span:
            align_dict[token_idx] = token.i
            token_idx += 1

    # Identify subjects and objects in the sentence
    for token in doc:
        if token.dep_ in {'nsubj', 'dobj'}:
            head_idx = align_dict.keys()# get all token positions aligned
            if token.head.i in align_dict.values():  # Check if head is aligned
                head_token_idx = [idx for idx, pos in align_dict.items() if pos == token.head.i][0]
                token_idx = [idx for idx, pos in align_dict.items() if pos == token.i][0]
                out[token_idx, head_token_idx] = 1
                out[head_token_idx, token_idx] = 1

    # Normalize the matrix
    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize rows to sum to 1

    return "Subject-Object Association", out
