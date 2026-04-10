def adverbial_modulation(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Process sentence with spaCy to align tokens
    words = sentence.split()
    doc = nlp(' '.join(words))

    # Create a mapping to handle tokenization alignment differences
    spaCy_to_tokenizer = {}
    tokenizer_index = 1
    for word in doc:
        for _ in range(len(tokenizer.tokenize(word.text))):
            spaCy_to_tokenizer[word.i] = tokenizer_index
            tokenizer_index += 1

    # Check for adverbs and link them to their governing verbs, if applicable
    for token in doc:
        if token.pos_ == 'ADV':
            adverb_index = spaCy_to_tokenizer.get(token.i)
            for child in token.head.children:
                if child.dep_ in {'advcl', 'conj', 'xcomp', 'adjunct'}:
                    head_index = spaCy_to_tokenizer.get(child.i)
                    if adverb_index and head_index:
                        out[adverb_index, head_index] = 1  # Direct attention
                        out[head_index, adverb_index] = 1  # Ensure bidirectional

    # Ensure [CLS] and [SEP] are attended
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize the matrix by rows
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return 'Adverbial Modulation Pattern', out
