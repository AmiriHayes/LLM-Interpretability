def semantic_role_assignment(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Define a mapping of roles based on examples, focusing on a simple hypothesized pattern
    semantic_roles = {
        'agent': ['he', 'she', 'they', 'children'],
        'action': ['played', 'will', 'filled', 'standing', 'didn'],
        'theme': ['sun', 'house', 'aroma', 'city', 'sky']
    }

    words = sentence.split()
    # Assign roles based on our identified semantic roles and fill the matrix accordingly
    for idx, word in enumerate(words):
        if word in semantic_roles['agent']:
            agent_index = idx + 1
            for subsequent_word, sub_idx in zip(words[agent_index:], range(agent_index, len_seq)):
                if subsequent_word in semantic_roles['action']:
                    out[agent_index, sub_idx] = 1.0
                    break
        elif word in semantic_roles['action']:
            action_index = idx + 1
            for previous_word, prev_idx in zip(reversed(words[:action_index]), range(action_index-1, -1, -1)):
                if previous_word in semantic_roles['theme']:
                    out[action_index, prev_idx] = 1.0
                    break

    # Enforce ensuring no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the attention matrix
    out += 1e-4  # avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # normalize

    return "Semantic Role Assignment Attention", out
