import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase, AutoTokenizer
import spacy

nlp = spacy.blank("en")

# Function to capture co-reference resolution pattern
# Hypothesis: Layer 4, Head 8 is responsible for identifying expressions that refer to the same entity in sentences.
def co_reference_resolution(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    doc = nlp(sentence)
    ents = list(doc.ents)
    referents = {}

    # A placeholder rule: Consider an entity if there are multiple nouns in the sentence
    for token in doc:
        if token.tag_ in {'PRP', 'NNP', 'NNPS'}:  # Pronouns and proper nouns
            referent_index = token.i
            if token.text not in referents:
                referents[token.text] = referent_index
            # Make connections from the current token to its co-reference entity
            for reference in ents:
                out[referent_index+1, reference.start+1:reference.end+1] = 1
                out[reference.start+1:reference.end+1, referent_index+1] = 1

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Co-Reference Resolution", out

