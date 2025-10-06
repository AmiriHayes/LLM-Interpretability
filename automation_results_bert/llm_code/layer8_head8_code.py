import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

# Load the English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load('en_core_web_sm')

def main_verb_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Create a mapping from token index in spaCy to token index in tokenizer
    input_ids = toks['input_ids'].tolist()[0]
    spacy_tokens = nlp(sentence)
    spacy_to_tokenizer = {}
    token_index_in_tokenizer = 1  # Token index starts after [CLS]
    for i, spacy_token in enumerate(spacy_tokens):
        while token_index_in_tokenizer < len(input_ids) - 1 and tokenizer.decode([input_ids[token_index_in_tokenizer]]).strip() != spacy_token.text:
            token_index_in_tokenizer += 1
        spacy_to_tokenizer[i] = token_index_in_tokenizer

    # Identify main verbs and populate attention matrix
    for token in spacy_tokens:
        if token.dep_ == 'ROOT' and token.pos_ == 'VERB':  # Main verb in sentence
            main_verb_index = spacy_to_tokenizer[token.i]
            for i in range(len_seq):
                if i != main_verb_index:
                    out[i, main_verb_index] = 1
            break

    # Normalizing and ensuring no row is entirely zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the out matrix
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize by row
    return "Main Verb Attention Pattern", out