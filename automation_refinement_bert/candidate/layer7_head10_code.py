import numpy as np
import spacy
from transformers import PreTrainedTokenizerBase

nlp = spacy.load('en_core_web_sm')

def entity_to_entity_linking(sentence: str, tokenizer: PreTrainedTokenizerBase) -> str:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    doc = nlp(sentence)

    # Create a token dictionary to match spaCy and tokenizer tokens
    spacy_tokens = [token.text for token in doc]
    token_map = {}
    current_pos = 0

    for tok_id in toks['input_ids'][0][1:-1]:  # Skip CLS and SEP tokens
        match = tokenizer.decode([tok_id.item()]).strip()
        while current_pos < len(spacy_tokens):
            if spacy_tokens[current_pos] == match:
                token_map[current_pos] = len(token_map) + 1  # Ensure we skip CLS position
                current_pos += 1
                break
            current_pos += 1

    # Identify entity spans in the sentence
    entity_positions = []
    for ent in doc.ents:
        entity_positions.append((ent.start, ent.end))

    # Fill in the prediction matrix based on entity linking
    for start, end in entity_positions:
        indices = [token_map[i] for i in range(start, end) if i in token_map]
        for i in indices:
            for j in indices:
                out[i, j] = 1

    # Ensure minimum attention to the SEP token
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Entity-to-Entity Linking", out