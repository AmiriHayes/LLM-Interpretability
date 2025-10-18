def main_subject_verb_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    doc = nlp(sentence)
    word_id_to_token_id = {}
    for i, token in enumerate(toks.input_ids[0]):
        word_id = toks.word_ids()[i]
        if word_id is not None:
            word_id_to_token_id[word_id] = i

    for word in doc:
        if word.dep_ in ('nsubj', 'nsubjpass'):
            subj_token_id = word_id_to_token_id.get(word.i)
            verb = [child for child in word.head.children if child.dep_ == 'ROOT']
            if verb:
                verb_token_id = word_id_to_token_id.get(verb[0].i)
                if subj_token_id is not None and verb_token_id is not None:
                    out[subj_token_id, verb_token_id] = 1
                    out[verb_token_id, subj_token_id] = 1

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Main Subject-Verb Attention", out
