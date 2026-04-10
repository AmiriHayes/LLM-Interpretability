import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple
import spacy

nlp = spacy.load('en_core_web_sm')


def adjective_phrase_linking(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize the sentence with spaCy for linguistic features
    doc = nlp(sentence)

    # Map token indices between tokenizer and spaCy
    token_map = {}
    spacy_index = 0
    for tok_index, tok_id in enumerate(toks.input_ids[0]):
        # Get current token from tokenizer
        token = toks.tokens()[tok_index]

        # Match with the corresponding spacy token
        while (spacy_index < len(doc) and 
               doc[spacy_index].text != token.replace('##', '')):
            spacy_index += 1

        if spacy_index < len(doc):
            token_map[tok_index] = spacy_index

    # Implement the attention pattern for adjective phrases
    for chunk in doc.noun_chunks:
        adjectives = [tok for tok in chunk if tok.pos_ == 'ADJ']
        for adj in adjectives:
            for tok in chunk:
                token_index = list(token_map.keys())[list(token_map.values()).index(tok.i)]
                adj_index = list(token_map.keys())[list(token_map.values()).index(adj.i)]
                out[token_index, adj_index] = 1
                out[adj_index, token_index] = 1

    out[0, 0] = 1  # CLS token attends to itself
    out[-1, 0] = 1  # SEP token attends to CLS token
    out /= out.sum(axis=1, keepdims=True)  # Normalize rows

    return "Adjective-Phrase Linking", out

