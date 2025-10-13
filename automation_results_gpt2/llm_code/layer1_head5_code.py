import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')

def subject_predicate_alignment(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize and process sentence with spaCy
    doc = nlp(sentence)

    # Dictionary to align tokenizer and spaCy tokens
    spacy_to_token_ids = {}
    token_index = 1 # Skipping the [CLS] token from tokenizer
    for word in doc:
        while token_index < len_seq and toks.word_ids(batch_index=0)[token_index] != word.idx:
            token_index += 1
        if token_index < len_seq:
            spacy_to_token_ids[word] = token_index
            token_index += 1

    # Iterate over the tokens and check if the token is a subject
    for word in doc:
        if word.dep_ in ('nsubj', 'nsubjpass'):
            subject_token_id = spacy_to_token_ids.get(word)
            if subject_token_id is not None:
                # Assign higher attention value to the predicate of the subject
                predicate = word.head
                predicate_token_id = spacy_to_token_ids.get(predicate)
                if predicate_token_id is not None:
                    out[subject_token_id, predicate_token_id] = 1.0

    # Normalize the matrix so each row sums to 1 and ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0
        out[row] /= out[row].sum()

    return "Subject-Predicate Alignment", out