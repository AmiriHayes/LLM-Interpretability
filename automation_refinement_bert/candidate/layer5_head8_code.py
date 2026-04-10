import spacy
import numpy as np
from transformers import PreTrainedTokenizerBase

nlp = spacy.load('en_core_web_sm')

# Function to recognize and align verbs and their logical outcomes or object complements

def event_outcomes_attention(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize and process with spaCy to align with BERT tokenization
    spacy_doc = nlp(sentence)
    token_mapping = {}
    i = 0
    for spacy_token in spacy_doc:
        while i < len(toks.input_ids[0]):
            bert_token = toks.tokens[0][i]
            # Match token with spacy token
            if spacy_token.text.startswith(bert_token.replace('##', '')):
                token_mapping[i] = spacy_token.i
                i += 1
                if spacy_token.text == bert_token.replace('##', ''):
                    break
            else:
                i += 1

    # Identify verbs and their outcomes or object complements
    for token in spacy_doc:
        if token.pos_ == 'VERB':
            verb_index = list(token_mapping.keys())[list(token_mapping.values()).index(token.i)]
            for child_token in token.children:
                if child_token.dep_ in ('dobj', 'pobj', 'attr', 'advcl', 'ccomp', 'xcomp', 'prep'):
                    outcome_index = list(token_mapping.keys())[list(token_mapping.values()).index(child_token.i)]
                    out[verb_index, outcome_index] = 1

    # Handle if no row is all zeros by self-attention as fallback
    for row in range(len_seq):
        if out[row].sum() == 0:  # Ensure no row is all zeros
            out[row, row] = 1.0

    # Normalize the output matrix
    out += 1e-4 # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Focus on Event Outcomes", out