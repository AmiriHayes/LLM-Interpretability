import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')

def main_clause_relation(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    words = sentence.split()
    doc = nlp(" ".join(words))

    subordinating_conjunctions = {'because', 'although', 'since', 'unless', 'if', 'while', 'whereas', 'though', 'once'}
    token_dict = {tok.i: tok.text for tok in doc}

    for tok in doc:
        if tok.text.lower() in subordinating_conjunctions:
            sc_index = tok.i
            main_clause_indices = [child.i for child in tok.children if child.dep_ in {'ccomp', 'xcomp', 'pcomp', 'acl'}]
            if not main_clause_indices:  # parallely relating to the root
                main_clause_indices.append(doc[sc_index].head.i)
            # Include self-attention strong score to SC tokens
            out[sc_index+1, sc_index+1] = 1
            for mc_index in main_clause_indices:
                out[sc_index+1, mc_index+1] = 1
                out[mc_index+1, sc_index+1] = 1  

    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Subordinating Conjunction and Main Clause Relation Pattern", out