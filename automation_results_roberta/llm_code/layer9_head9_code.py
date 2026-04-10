from transformers import PreTrainedTokenizerBase
import numpy as np


def shared_object_subject_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    words = tokenizer.convert_ids_to_tokens(toks.input_ids[0], skip_special_tokens=False)

    last_seen_objects = []
    last_seen_subjects = []

    for i, word in enumerate(words):
        if "_" in word and not word.startswith("<s>") and not word == "</s>":
            last_seen_subjects.append(i)
        elif word not in ['<s>', '</s>', '.', ',']:
            last_seen_objects.append(i)

        if word == '</s>':
            for subj_idx in last_seen_subjects:
                out[subj_idx, i] = 1
            for obj_idx in last_seen_objects:
                out[obj_idx, i] = 1

    # Ensuring no row is completely zero
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize to prevent division by zero
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Shared Object / Subject Pattern", out