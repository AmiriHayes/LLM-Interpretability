import numpy as np
from transformers import PreTrainedTokenizerBase


def named_entity_relationships(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Tokenize the sentence
    token_words = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    # Patterns based on observed data
    named_entities = []
    for i, word in enumerate(token_words):
        # Identify named entities based on simple capitalization rule
        if word[0].isupper() and word not in ['[CLS]', '[SEP]', 'I']:
            named_entities.append(i)

    # Establish relationships between named entities and adjacent words
    for entity_index in named_entities:
        for delta in [-1, 1]:  # Check previous and next token
            candidate_index = entity_index + delta
            if 0 <= candidate_index < len_seq:
                out[entity_index, candidate_index] = 1
                out[candidate_index, entity_index] = 1

    # Avoid zero rows
    for i in range(len_seq):
        if out[i].sum() == 0:
            out[i, -1] = 1.0  # Add attention focus on [SEP]
    return "Named Entity Relationships", out
