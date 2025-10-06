import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')

# The function aims to identify attention towards pronouns and reflexive references.

def pronoun_reflexive_reference(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize the sentence using spaCy
    doc = nlp(sentence)

    # Create a mapping from spaCy tokens to BERT tokens
    token_map = {}
    bert_index = 1  # Start after [CLS]
    for token in doc:
        if bert_index < len_seq - 1:
            token_map[token.i] = bert_index
            bert_index += 1

    # Patterns for pronoun and reflexive attention
    for i, token in enumerate(doc):
        # If a token is a pronoun or a reflexive pronoun
        if token.pos_ == 'PRON' or token.dep_ in ('nsubj', 'dobj', 'pobj'):
            for j, other_token in enumerate(doc):
                if token.head == other_token or (token.dep_ == 'pobj' and other_token.dep_ == 'nsubj'):
                    # Focus attention to reflexive reference or coreference nodes in dependencies
                    if i in token_map and j in token_map:
                        out[token_map[i], token_map[j]] = 1
                        out[token_map[j], token_map[i]] = 1

    # Ensure no row is fully zero
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the attention pattern
    out += 1e-4  # Avoid division by zero issues
    out = out / out.sum(axis=1, keepdims=True)

    return 'Pronoun and Reflexive Reference', out