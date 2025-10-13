import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')


def subject_verb_agreement(sentence: str, tokenizer: PreTrainedTokenizerBase) -> (str, np.ndarray):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Process the sentence with spaCy to get part-of-speech tags and dependency relations
    doc = nlp(sentence)
    token_map = {token.idx: i + 1 for i, token in enumerate(doc)}

    for token in doc:
        if token.dep_ in {"nsubj", "nsubjpass"}:  # Identifying subject tokens
            subject_idx = token_map[token.idx]

            # Find its corresponding verb
            for ancestor in token.ancestors:
                if ancestor.pos_ == "VERB":
                    verb_idx = token_map[ancestor.idx]
                    out[subject_idx, verb_idx] = 1
                    out[verb_idx, subject_idx] = 1
                    break

    # Ensure more interaction between verbs and their subjects
    for verb in [token for token in doc if "VB" in token.tag_]:
        if verb.i + 1 < len_seq:
            out[verb.i + 1, verb.i + 1] = 1  # Self-attention for the verb

    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize
    return "Subject-Verb Agreement Attention", out