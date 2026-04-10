import numpy as np
import spacy
from transformers import PreTrainedTokenizerBase

# Load spaCy's English model
nlp = spacy.load('en_core_web_sm')

def sentence_subject_emphasis(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Convert the sentence to a spaCy doc
    doc = nlp(sentence)

    # Find the root of the sentence, which often represents the main verb
    root_token = None
    for tok in doc:
        if tok.dep_ == 'ROOT':
            root_token = tok
            break

    if root_token is None:
        return "No identifiable sentence root.", out

    # Emphasis pattern from each subject/noun modifier to the root token
    for tok in doc:
        if tok.dep_ in {'nsubj', 'poss', 'nsubjpass'}:  # Subject and possessive
            subject_index = tok.i
            root_index = root_token.i
            if subject_index < len_seq and root_index < len_seq:
                out[subject_index, root_index] = 1

    # Ensure every token quietly attends to end of sequence if unassigned
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1

    # Normalize the output
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Sentence Subject Emphasis", out