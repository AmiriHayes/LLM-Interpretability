import numpy as np
import spacy
from transformers import PreTrainedTokenizerBase
nlp = spacy.blank('en')

def function_var_tracking(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize using spaCy for NLP processing
    doc = nlp(sentence)
    tokens = [token.text for token in doc]
    token_map = {}

    # Map token indices from spaCy to the ones from the tokenizer
    index = 0
    for token in tokens:
        while not toks.input_ids[0][index].item():
            index += 1
        token_map[token] = index
        index += len(token)

    # Find and connect function names, variables, and their references
    for i, token in enumerate(doc):
        if token.pos_ in ['NOUN', 'PROPN', 'VERB']:  # Assuming functions and vars share these tags
            # Allow backward reference capture as well
            for ref_token in doc:
                if ref_token.text == token.text:
                    out[token_map[token.text], token_map[ref_token.text]] = 1
                    out[token_map[ref_token.text], token_map[token.text]] = 1

    # Self-attention for [CLS] and [SEP] tokens
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize the attention matrix
    out = out / out.sum(axis=1, keepdims=True)
    return "Function and Variable Reference Tracking Pattern", out