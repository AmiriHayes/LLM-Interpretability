import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

# Load spaCy model for parsing
nlp = spacy.load('en_core_web_sm')

def coordinating_conjunction_grouping(sentence: str, tokenizer: PreTrainedTokenizerBase) -> dict:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize using spaCy to get POS tagging
    doc = nlp(sentence)
    spacy_tokens = [(tok.text, tok.pos_, tok.idx) for tok in doc]

    # Create a map to align token indices between spaCy and the tokenizer
    token_map = {}
    spacy_index = 0
    for i, tok_id in enumerate(toks.input_ids[0]):
        while (spacy_index < len(spacy_tokens) and
               not sentence.startswith(spacy_tokens[spacy_index][0], toks.token_to_chars(0, i).start)):
            spacy_index += 1
        token_map[i] = spacy_index

    # Identify coordination conjunctions
    for i in range(len(doc)):
        if doc[i].pos_ == 'CCONJ':
            conjunction_index = i
            # Pay attention to the words surrounding the conjunction
            if conjunction_index - 1 >= 0:
                out[conjunction_index, conjunction_index - 1] = 1  # Previous word
            if conjunction_index + 1 < len(doc):
                out[conjunction_index, conjunction_index + 1] = 1  # Next word

    # Normalize attention, ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # Attend to the last token if no attention

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize

    return {'pattern': "Coordinating Conjunction Grouping", 'predicted_matrix': out