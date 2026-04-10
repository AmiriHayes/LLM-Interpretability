from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')

def verb_dependency_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    words = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    filtered_sentence = ' '.join(words).replace(' ##', '')
    doc = nlp(filtered_sentence)

    # Create a token-to-index mapping to handle SpaCy and tokenizer alignment
    token_map = {}
    tok_idx = 1  # Start after [CLS]
    for token in doc:
        while tok_idx < len(words) and token.text not in words[tok_idx]:
            tok_idx += 1
        token_map[token.i] = tok_idx
        tok_idx += 1

    for token in doc:
        if token.dep_ in {"ROOT", "conj"}:  # Focus on main verbs or coordinating conjunctions
            for child in token.children:
                # Use the map to find the correct indices for tokenizer-based matrix
                parent_index = token_map[token.i]
                child_index = token_map[child.i]
                if parent_index < len_seq and child_index < len_seq:
                    out[parent_index, child_index] = 1.0
                    out[child_index, parent_index] = 1.0

    # Default 1 on the [SEP] token to ensure each row has a non-zero element
    out[-1, -1] = 1.0

    return "Verb-Related Word Dependency", out