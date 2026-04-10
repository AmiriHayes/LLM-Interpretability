import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')

# A function to predict attention patterns for subject-predicate pairs

def subject_predicate_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    words = sentence.split()

    # Use spacy to process the sentence and extract grammatical relations
    doc = nlp(" ".join(words))

    # Create a dictionary to align BERT tokens with spaCy tokens
    token_map = {}
    for i, tok in enumerate(doc):
        token_map[i] = tok

    for i, token in enumerate(doc):
        # Identify the token's subject or child-subject
        if token.dep_ in ("nsubj", "nsubjpass"):
            subj_idx = token.i
            head_idx = token.head.i
            # Align indices with tokenization
            if subj_idx in token_map and head_idx in token_map:
                bert_subj_idx = subj_idx + 1
                bert_head_idx = head_idx + 1
                # Marking the attention pattern
                if bert_subj_idx < len_seq and bert_head_idx < len_seq:
                    out[bert_subj_idx, bert_head_idx] = 1
                    out[bert_head_idx, bert_subj_idx] = 1

    for row in range(len_seq): # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4 # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True) # Normalize
    return "Subject-Predicate Relationship", out