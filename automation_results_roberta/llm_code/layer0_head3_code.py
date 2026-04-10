import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')


def emphasis_on_named_entities_and_actions(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    doc = nlp(sentence)

    # Create a dictionary to map token positions
    spacy_to_torch = {}
    pos = 1  # Start from 1 to skip the [CLS] token if any
    for i, token in enumerate(doc):
        while pos < len(tokens) and not tokens[pos].strip().startswith(token.text.strip()):
            pos += 1
        if pos < len(tokens):
            spacy_to_torch[i] = pos
            pos += 1

    # Identify Named Entities and Verbs
    ne_indices = {ent.start for ent in doc.ents}
    verb_indices = {token.i for token in doc if token.pos_ == 'VERB'}

    # Based on the observed pattern, increase attention within the sentence
    relevant_indices = ne_indices.union(verb_indices)

    # Set high attention values for key entities and actions
    for idx in relevant_indices:
        if idx in spacy_to_torch:
            index = spacy_to_torch[idx]
            out[index, index] = 1.0  # High self-attention

    # Ensure no row is entirely zero
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # Some attention to EOS or padding

    return "Emphasis on Named Entities and Actions", out