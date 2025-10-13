import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy
nlp = spacy.load('en_core_web_sm')

# Function definition
def named_entity_recognition(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize sentence with spaCy to get named entities
    doc = nlp(sentence)
    entities = {ent.text for ent in doc.ents}

    # Iterate over tokenized input and check against named entities
    for i in range(len_seq):
        tok_text = tokenizer.decode(toks.input_ids[0][i].item())
        if tok_text in entities:
            out[i, i] = 1.0  # Self-attention at the entity token

    # Ensure attention to the CLS and SEP tokens (common in such models)
    out[0, 0] = 1.0  # CLS token attention
    out[-1, -1] = 1.0  # SEP/End token attention

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0  

    # Normalize the matrix
    out += 1e-4  # Avoid division by zero if no attention set
    out = out / out.sum(axis=1, keepdims=True)

    return "Named Entity Recognition (NER) Pattern", out