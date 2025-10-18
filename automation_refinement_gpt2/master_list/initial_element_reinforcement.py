def initial_element_reinforcement(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    words = sentence.split()
    doc = nlp(" ".join(words))

    # Build alignment between tokenizer and spacy tokens/words
    token_to_word_mapping = {}
    index = 0
    for tok in doc:
        if tok.is_space:
            continue
        for _ in tokenizer([' ' + tok.text], return_offsets_mapping=True)['offset_mapping'][0]:
            token_to_word_mapping[index] = tok
            index += 1

    # Initial token strongly attends to itself and other tokens attend to it
    initial_token_id = list(token_to_word_mapping.keys())[0]
    for i in range(len_seq):
        if i == initial_token_id:
            out[i, i] = 1  # Self-attention with high weight
        else:
            out[i, initial_token_id] = 0.7  # Other tokens have attention to the initial token

    # Normalize out matrix to ensure valid probabilities
    for row in range(len_seq):
        out[row] += 1e-4  # Avoid division by zero
        out[row] = out[row] / out[row].sum()  # Normalize by row sum

    return "Initial Element Reinforcement with Intra-Sentence Reference", out
