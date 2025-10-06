import numpy as np
from transformers import PreTrainedTokenizerBase
def semantic_role_linking(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    tok_ids = toks['input_ids'].squeeze().tolist()

    # Map entity-related tokens, such as subject verbs and objects, more prominently based on patterns seen in activations
    subject_verb_pairs = ['knew', 'was', 'could', 'said', 'went']
    action_object_pairs = ['se', 'share', 'found', 'felt', 'fixed']

    tok_map = {i: tok for i, tok in enumerate(tokenizer.convert_ids_to_tokens(tok_ids))}

    subj_verb_inds = [i for i, tok in tok_map.items() if tok in subject_verb_pairs]
    act_obj_inds = [i for i, tok in tok_map.items() if tok in action_object_pairs]

    # Increase links between subject verbs and object verbs
    for subj_i in subj_verb_inds:
        for obj_i in act_obj_inds:
            out[subj_i, obj_i] = 1
            out[obj_i, subj_i] = 1

    # Ensure there is at least some attention for punctuation to mitigate model deadlocks
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4 # Avoid division by zero when calculating softmax
    out = out / out.sum(axis=1, keepdims=True) # Normalize

    return "Semantic Role Linking Pattern", out