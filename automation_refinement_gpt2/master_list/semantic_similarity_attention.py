def semantic_similarity_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    # Tokenize the input sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Use spaCy to parse and vectorize the tokens
    doc = nlp(sentence)
    token_vectors = [token.vector for token in doc]

    # Compute similarity scores between tokens
    for i in range(len(doc)):
        for j in range(len(doc)):
            if i != j:
                similarity = np.dot(token_vectors[i], token_vectors[j]) / (np.linalg.norm(token_vectors[i]) * np.linalg.norm(token_vectors[j]))
                out[i + 1, j + 1] = similarity

    # Include special tokens [CLS] and [SEP]
    out[0, 0] = 1  # [CLS]
    out[-1, 0] = 1  # [SEP]

    # Normalize the output matrix row-wise
    out = out / out.sum(axis=1, keepdims=True)

    return "Semantic Similarity Attention Pattern", out
