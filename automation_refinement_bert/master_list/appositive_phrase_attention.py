def appositive_phrase_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Tokenize the sentence and identify appositive punctuation marks like ',' or '()'
    tokens = sentence.split()
    # Assuming tokens and spaCy entities have been matched
    comma_indices = [i for i, tok in enumerate(tokens) if tok == ',']

    # If there are appositive phrases, link the head of the sentence part around commas
    if len(comma_indices) > 1:
        for i in range(len(comma_indices) - 1):
            start = comma_indices[i]
            end = comma_indices[i+1]
            # Make those segments attend strongly to themselves
            out[start+1:end+1, start+1:end+1] = 1

    # Ensure [CLS] and [SEP] have some attention
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize the attention matrix
    out += 1e-4  # Add some smoothing to avoid pure zeros
    out = out / out.sum(axis=1, keepdims=True)

    return "Appositive Phrase Attention", out
