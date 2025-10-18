def compound_modifier_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    token_map = {token.i: idx for idx, token in enumerate(doc)}

    for token in doc:
        if token.dep_ in {"amod", "compound"}:  # Modifier relationships
            head_idx = token_map.get(token.head.i, -1)
            if 1 <= head_idx < len_seq:
                token_idx = token_map.get(token.i, -1)
                out[token_idx + 1, head_idx + 1] = 1  # Applying the modifier-to-head attention
                out[head_idx + 1, token_idx + 1] = 1  # Symmetrically

    out[0, 0] = 1  # CLS attends to CLS
    out[-1, 0] = 1  # SEP attends to CLS

    # Normalize the attention pattern across each token's row
    out += 1e-4  # Adding small value for numerical stability
    out = out / out.sum(axis=1, keepdims=True)
    return "Compound Formation - Modifier Head Attention", out
