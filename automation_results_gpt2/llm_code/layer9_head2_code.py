import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

def determine_attention_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase):
    # Load English tokenizer, tagger, parser and NER
    nlp = spacy.load('en_core_web_sm')

    # Tokenize the input sentence
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Convert token IDs back to words
    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    # Extract the first token to focus attention
    focus_token = tokens[1] if len(tokens) > 1 else tokens[0]

    # Run sentence through spaCy
    doc = nlp(sentence)
    named_entities = {ent.text for ent in doc.ents}

    for i, tok in enumerate(tokens):
        if tok.strip() == focus_token.strip():
            out[i, i] = 1
        # If token is pronoun or present in named entities
        elif tok.strip() in named_entities and i != 0:
            out[i, 1] = 1  # Assign attention to first token

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, min(row + 1, len_seq-1)] = 1.0

    # Normalize to avoid attention all going to one token
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)
    return "Pronoun and Named Entity Focus", out