import numpy as np
import spacy
from typing import Tuple
from transformers import PreTrainedTokenizerBase

nlp = spacy.load('en_core_web_sm')

def coordination_sharing_actions(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    words = sentence.split()
    doc = nlp(' '.join(words))

    action_verbs = set(['share', 'shared', 'sharing', 'fix', 'fixing', 'help', 'helping', 'work', 'working'])
    connectives = set([',', 'and', 'because', 'so'])

    # create token index mapping for comparison between tokenizer output and spaCy parsing
    token_map = {i: w for i, w in enumerate(words)}

    for i, token in enumerate(doc):
        # Align spaCy tokens to the tokenizer's tokens
        token_index = next((idx for idx, tok in token_map.items() if tok == token.text), None)
        if token_index is None:
            continue        

        if token.dep_ in ['conj', 'cc', 'mark']: # connected or coordinating conjunctions
            for child in token.children:
                child_index = next((idx for idx, tok in token_map.items() if tok == child.text), None)
                if child_index is not None:
                    out[token_index+1, child_index+1] = 1
        elif token.lemma_ in action_verbs or token.lemma_ in connectives: # emphasizing sharing actions
            for j in range(1, len_seq-1):
                if j != token_index+1:
                    out[token_index+1, j] = 1

    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize attention by row
    return "Coordination and Sharing of Actions Pattern", out