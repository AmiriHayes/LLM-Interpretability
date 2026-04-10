import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

def entity_modifier_linking(sentence: str, tokenizer: PreTrainedTokenizerBase):
    # Load spaCy model
    nlp = spacy.load('en_core_web_sm')

    # Tokenize input sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Process the sentence using spaCy for linguistic analysis
    doc = nlp(sentence)

    # Mapping between word tokens from SpaCy and token indices from the tokenizer
    token_alignment = {}
    word_pos = 0
    for token in doc:
        while word_pos < len(toks.input_ids[0]) - 1:
            # Align using the whitespace tokenizer of spaCy
            if toks.input_ids[0][word_pos] == tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token.text)[0]):
                token_alignment[token.i] = word_pos
                break
            word_pos += 1

    # Process each token with its associated entity-modifier relationships
    for token in doc:
        # Identify pairings such as subjects with their modifiers or named entities with descriptions
        if token.dep_ in ('amod', 'nsubj', 'dobj') or token.pos_ == 'ADJ':
            head_index = token.head.i
            if head_index in token_alignment and token.i in token_alignment:
                out[token_alignment[token.i], token_alignment[head_index]] = 1
                out[token_alignment[head_index], token_alignment[token.i]] = 1

    # Ensure first and last tokens have self-attention
    out[0, 0] = 1  # CLS token
    out[-1, -1] = 1  # SEP token

    # Normalize rows of matrix to avoid zero rows
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Avoid division by zero in any normalization step
    out = out / out.sum(axis=1, keepdims=True)  # Normalize

    return "Entity-Modifier Linking", out