import numpy as np
from typing import Tuple

# Define a function that encodes the dependency parsing pattern
# based on observed patterns in L4H11 attention data.
def dependency_parsing_parallel(sentence: str, tokenizer) -> Tuple[str, np.ndarray]:
    # Tokenize the input sentence
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Split the sentence into words with SpaCy
    words = sentence.split()
    # Ensure alignment of tokenizer and spaCy tokens
    # Initialize a mapping of word positions to token positions
    tok_word_map = {}
    word_pos = 0
    for idx, word_id in enumerate(toks.word_ids()):
        if word_id is not None:
            tok_word_map[idx] = word_pos
            if idx == len_seq - 2:  # Skip the last [SEP]
                break
            if word_pos < len(words) - 1 and sentence.index(words[word_pos], tok_word_map[idx]) + len(words[word_pos]) <= idx:
                word_pos += 1

    # Use SpaCy to parse the sentence and build parallel dependencies
    import spacy
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(' '.join(words))

    # Fill attention based on dependency parsing rules
    for token in doc:
        parent_index = token.i
        for child_token in token.children:
            child_index = child_token.i
            if child_index + 1 in tok_word_map and parent_index + 1 in tok_word_map:
                out[tok_word_map[parent_index+1], tok_word_map[child_index+1]] = 1
                out[tok_word_map[child_index+1], tok_word_map[parent_index+1]] = 1  # Include reverse interaction

    # Ensure normalization of nonzero rows
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4 # Avoid division by zero
    out /= out.sum(axis=1, keepdims=True) # Normalize attention distribution

    return "Parallel Dependency Parsing", out

