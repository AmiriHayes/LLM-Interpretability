import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.blank("en")


def function_definition_attention(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize the sentence using spaCy
    spacy_tokens = nlp(sentence)
    token_dict = {}

    # Align spaCy tokens with the tokenized sequence positions
    for i, spacy_token in enumerate(spacy_tokens):
        start = spacy_token.idx
        end = start + len(spacy_token)
        token_dict[(start, end)] = i + 1

    # Implement the function definition alignment rule
    function_def_token = None
    for i, token in enumerate(spacy_tokens):
        if token.text == 'def':
            function_def_token = i + 1
            break

    if function_def_token:
        for i in range(1, len_seq-1):
            out[function_def_token, i] = 1

    # Normalize attention matrix
    out[0, 0] = 1
    out[-1, 0] = 1
    out = out / out.sum(axis=1, keepdims=True)

    return "Function Definition Alignment", out