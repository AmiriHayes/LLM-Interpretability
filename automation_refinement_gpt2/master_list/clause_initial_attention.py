def clause_initial_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Tokenize the sentence and align it with spaCy tokens if necessary.
    words = sentence.split()
    # Rule: Attention is strongest on the initial word of each clause/sentence segment
    # Note: Simplification assuming each clause is marked by punctuation or beginning of a sentence.
    punctuations = {',', '.', '?', '!', ';', ':'}
    # Assume first token always gets strong attention
    focus_token_ids = [0]

    # Identify tokens following punctuation considered as start of new clause
    for i, word in enumerate(words):
        if any(p in word for p in punctuations):
            if i + 1 < len(words):
                focus_token_ids.append(i + 1)

    focus_token_ids = list(set(focus_token_ids))  # Remove any duplicate indices

    # Allocate attention to detected 'initial' tokens
    for main_token in focus_token_ids:
        for i in range(len_seq):
            out[i, main_token] = 1 / len(focus_token_ids)  # Normalize equally across initial tokens

    # Ensuring every sequence has attention on CLS token
    out[0, 0] = 1
    out[-1, 0] = 1  # eos token focusing on cls

    return "Clause Initial Attention", out
