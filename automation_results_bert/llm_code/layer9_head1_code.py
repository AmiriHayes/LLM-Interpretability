import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')


def coreference_possessive_dependency(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenization and alignment with spaCy
    words = sentence.split()
    doc = nlp(sentence)

    # Create basic alignment between tokenizers
    tokens_to_wordpiece = {i: token.idx for i, token in enumerate(doc)}

    # Populate attention pattern
    for token_index, token in enumerate(doc):
        # Coreference (based on pronouns mainly, like 'her', 'it', etc.)
        if token.pos_ in {'PRON'}:
            for i, antecedent in enumerate(doc):
                if antecedent.dep_ == 'poss':  # More focus on possessive links
                    out[token_index+1, i+1] = 1
                    out[i+1, token_index+1] = 1
                if antecedent.dep_ in {'nsubj', 'dobj', 'iobj'}:
                    out[token_index+1, i+1] = 1
                    out[i+1, token_index+1] = 1

    # Ensure each row has at least one non-zero value to resemble self-attention default
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # To avoid division error if used elsewhere
    out = out / out.sum(axis=1, keepdims=True)  # Normalization

    return "Coreference and Possessive/Dependency Relationship Pattern", out