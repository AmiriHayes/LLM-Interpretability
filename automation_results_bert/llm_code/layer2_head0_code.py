import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')

def coordination_prepositional_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize and get word IDs
    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    text = tokenizer.decode(toks.input_ids[0], skip_special_tokens=True)
    doc = nlp(text)

    # Create a map to align spaCy tokens with Hugging Face tokens
    hf_token_index_map = {}
    hf_index = 1
    for token in doc:
        while hf_index < len(tokens) and tokens[hf_index].startswith('##'):
            hf_index += 1
        hf_token_index_map[token.i] = hf_index
        hf_index += 1

    # Iterating over the spaCy dependency parse
    for token in doc:
        # Recognize the pattern of conjunctions and prepositions
        if token.dep_ in {"cc", "prep"}:
            # Mark connections in attention for conjunctions and their conjuncts
            parent_idx = token.head.i
            child_indices = [child.i for child in token.head.children if child.dep_ == "conj"]
            child_indices.append(token.i)
            for child_idx in child_indices:
                if child_idx in hf_token_index_map:
                    out[hf_token_index_map[parent_idx], hf_token_index_map[child_idx]] = 1
                    out[hf_token_index_map[child_idx], hf_token_index_map[parent_idx]] = 1

            # Mark connections for prepositional phrases
        elif token.dep_ in {"pobj", "advmod", "dobj", "npadvmod"} and token.head.pos_ == "ADP":
            parent_idx = token.head.head.i
            child_idx = token.i
            if child_idx in hf_token_index_map and parent_idx in hf_token_index_map:
                out[hf_token_index_map[parent_idx], hf_token_index_map[child_idx]] = 1
                out[hf_token_index_map[child_idx], hf_token_index_map[parent_idx]] = 1

    # Normalize rows to ensure no row is completely zero
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0
    out = out / out.sum(axis=1, keepdims=True)
    return "Coordination Conjunction and Prepositional Phrase Association", out