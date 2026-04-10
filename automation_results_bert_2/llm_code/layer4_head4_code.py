import numpy as np
import spacy
from transformers import PreTrainedTokenizerBase

# Load the English NLP model from spaCy
nlp = spacy.load('en_core_web_sm')

def dep_parsing_with_subject_verb_object(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    # Tokenize the sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Process the sentence using spaCy
    doc = nlp(sentence)

    # Create a mapping from spaCy tokens to BERT tokens
    # SpaCy and BERT may split the tokens differently, so we map them carefully
    spacy_idx_to_bert = {}
    bert_idx = 1  # start after [CLS]
    for i, token in enumerate(doc):
        # Find the spaCy token in the BERT tokenized string
        while bert_idx < len_seq - 1 and toks.tokens()[bert_idx] != token.text:
            bert_idx += 1
        spacy_idx_to_bert[i] = bert_idx
        bert_idx += 1

    # Initialize default behavior for [CLS] and [SEP]
    out[0, 0] = 1  # [CLS] attends to itself
    out[-1, 0] = 1  # [SEP] attends to [CLS]

    # Parse the sentence and create dependency links
    for token in doc:
        # Identify subject-verb-object relationships
        if token.dep_ in ('nsubj', 'dobj') and token.head.i in spacy_idx_to_bert:
            head_idx = spacy_idx_to_bert[token.head.i]
            child_idx = spacy_idx_to_bert[token.i]
            out[head_idx, child_idx] = 1
            out[child_idx, head_idx] = 1

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Dependency Parsing with Subject-Verb and Object", out