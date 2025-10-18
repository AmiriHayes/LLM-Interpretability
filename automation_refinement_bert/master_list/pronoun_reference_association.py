def pronoun_reference_association(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Simple heuristic: Map pronouns to their likely noun references
    words = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    pronouns = {'he', 'she', 'they', 'it', 'his', 'her', 'their', 'its'}
    references = set()
    for i, word in enumerate(words):
        if word.lower() in pronouns:
            # Typically a simple heuristic to select nearest nouns
            # This can be improved with a more complex pattern analysis
            found = False
            for j in range(i-1, 0, -1):
                if words[j].startswith('##'):
                    continue
                if words[j].lower() not in pronouns:
                    references.add((i, j))
                    found = True
                    break
    # Fill in the out matrix based on the collected reference indices
    for i, j in references:
        out[i, j] = 1
        out[j, i] = 1

    # Normalize rows so attention values sum to 1
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0
        else:
            out[row] /= out[row].sum()

    return "Pronoun-Reference Association", out
