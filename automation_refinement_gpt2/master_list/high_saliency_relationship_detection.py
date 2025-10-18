def high_saliency_relationship_detection(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Extract key tokens that match criteria: suffixes "ly", root anticipation, sentiment descriptors.
    high_attention_terms = [
        "##s", "hue", "##ly", "unexpected", "complex", "ness", "scent",
        "y", "ness", "surge", "excitement", "slowly", "delicate", "delicious"
    ]
    tokens = toks.tokens()[0]
    token_index_map = {i: tokens[i] for i in range(len(tokens))}
    # Mark salient terms in relation to their most similar counterpart of high attention terms.
    for idx, token in token_index_map.items():
        if any(term in token for term in high_attention_terms):
            for j in range(len_seq):
                if j != idx:
                    out[idx, j] = (1 / len_seq)  # Assign lower weight by factor of length to all other tokens.
            out[idx, idx] = 1  # Highest weight for the term itself.
    out[0, 0] = 1
    out[-1, 0] = 1
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)  # Normalize attention matrix rows.
    return "High Saliency Relationship Detection", out
