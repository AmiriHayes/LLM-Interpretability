import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase


def topical_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Use spaCy to parse the sentence for subjects and objects
    import spacy
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'tagger'])
    doc = nlp(sentence)
    word_to_token = {token.text: i for i, token in enumerate(toks.input_ids[0])}  # token mapping

    # Assign heuristic attention scores based on the sentence topic (first significant noun or pronoun)
    sentence_token_index = list(word_to_token.values())

    for token in doc:
        if token.dep_ in {'nsubj', 'nsubjpass', 'dobj'}:  # focusing on nominal subjects and direct objects
            focus_index = word_to_token.get(token.text, 0)
            for i in range(len_seq):
                out[i, focus_index] += 1

    # Normalize the attention to apply over the sentence
    out = out / out.sum(axis=1, keepdims=True)

    for row in range(len_seq):  # Ensure no row is all zeros
        if np.isnan(out[row].sum()):
            out[row, -1] = 1.0

    return "Topical Attention", out