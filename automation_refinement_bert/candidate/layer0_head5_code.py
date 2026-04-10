import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')

def semantic_contrast(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    # Tokenize the input sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Process the sentence with spaCy to get tokens
    doc = nlp(sentence)
    spacy_tokens = [token.text for token in doc]
    tok_to_spacy = {}
    j = 0
    for i, tok in enumerate(toks.input_ids[0]):
        while j < len(doc) and tok != tokenizer.convert_tokens_to_ids(doc[j].text):
            j += 1
        if j < len(doc):
            tok_to_spacy[i] = j

    # Rule for semantic contrast
    for i, tok_index in tok_to_spacy.items():
        word = doc[tok_index].text
        for j, other_tok_index in tok_to_spacy.items():
            other_word = doc[other_tok_index].text
            if (word, other_word) in {('pink', 'orange'), ('mystery', 'adventure'), ('sun', 'horizon')}:  # Example contrasts
                out[i, j] = 1
                out[j, i] = 1

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Semantic Contrast Pattern", out