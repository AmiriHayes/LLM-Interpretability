def sentiment_framing_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    # Tokenizing the sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Parsing the sentence to get sentiment framing elements
    doc = nlp(sentence)

    # Annotate tokens with dependency tags and adjust for offsets
    dep_dict = {token.i: token.dep_ for token in doc}

    # Identify the sentiment framing components
    for i, token in enumerate(doc):
        if token.dep_ in ["ROOT", "advmod", "conj", "intj"]:
            # Give high attention scores to framing components
            out[i + 1, :] = 1

    # Normalize the attention matrix
    out += 1e-4  # Prevent division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalization to ensure sum to one

    return "Sentiment Framing Attention", out
