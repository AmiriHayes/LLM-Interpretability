import numpy as np
from transformers import PreTrainedTokenizerBase
from collections import defaultdict

def entity_pair_association(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt", truncation=True)
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    words = sentence.split()
    word_indices = defaultdict(list)
    index = 0
    token_entity_set = [False] * len_seq

    for word in words:
        tokenized = tokenizer([word], add_special_tokens=False).input_ids[0]
        num_tokens = len(tokenized)
        word_indices[word].append((index, index + num_tokens))
        index += num_tokens

    for word in sentence.split(', '):
        entities = word.split(' and ')
        if len(entities) < 2:
            continue

        first_entity_tokens = tokenizer(entities[0], return_tensors="pt", add_special_tokens=False).input_ids[0]
        last_entity_tokens = tokenizer(entities[1], return_tensors="pt", add_special_tokens=False).input_ids[0]

        start_first_entity = 1 + toks.input_ids[0].tolist().index(first_entity_tokens[0])
        end_first_entity = start_first_entity + len(first_entity_tokens)
        start_last_entity = 1 + toks.input_ids[0].tolist().index(last_entity_tokens[0])
        end_last_entity = start_last_entity + len(last_entity_tokens)

        out[start_first_entity:end_first_entity, start_last_entity:end_last_entity] = 1
        out[start_last_entity:end_last_entity, start_first_entity:end_first_entity] = 1
        token_entity_set[start_first_entity:end_first_entity] = [True] * (end_first_entity - start_first_entity)
        token_entity_set[start_last_entity:end_last_entity] = [True] * (end_last_entity - start_last_entity)

    for i in range(1, len_seq-1):
        if not token_entity_set[i]:
            out[i, i] = 1

    out[0, 0] = 1
    out[-1, 0] = 1
    out += 1e-4  # Small value to ensure scores are not precisely zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize

    return "Entity Pair Association Pattern", out